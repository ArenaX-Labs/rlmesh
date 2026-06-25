//! Bindings for the `rlmesh-adapters` core: spec resolution and plan
//! application.
//!
//! The env side is given as sparse tags (JSON) over the env's
//! observation/action spaces (gymnasium space objects, parsed here into the
//! adapters [`SpaceView`]); the model side is its declared spec (JSON).
//! [`adapters_resolve`] joins and resolves them into a plan handle.
//!
//! Values cross the boundary as the canonical Python RLMesh value tree:
//! `Tensor`, `str`, `bytes`, numbers, lists/tuples, and nested mappings.
//! Framework-specific conversion (NumPy/Torch/JAX) stays in Python; this binding
//! only bridges the canonical value tree into `rlmesh-adapters`.
//!
//! Custom inputs are never evaluated here: the plan keeps them as holes
//! ([`SkipCustoms`]) and the Python wrapper runs the user's callable on the
//! raw Python observation afterwards.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use rlmesh_adapters::v1::{
    ApplyError, CustomTransform, EncodingTransform, EnvTags, ModelInput, ModelSpec, ObsPlan,
    ResolvedAdapter, SkipCustoms, SpaceView, Value, join, resolve, roles,
};
use serde::de::DeserializeOwned;

/// Deserialize a spec JSON string with a field-path-annotated error.
///
/// `label` is the human spec name (`"env tags"` / `"model spec"`). The Rust
/// serde codec is the single authoritative validator; this surfaces its errors
/// to Python as clean, field-named messages like
/// `invalid model spec at action.components[0].dim: must be a non-negative
/// integer, got -1` instead of a bare line/column serde error.
///
/// Two documented limitations of the underlying `serde_path_to_error`:
/// - the field path stops at an internally-tagged-enum boundary, so a bad field
///   inside a `ModelInput`/`ObsTag` variant reports `inputs[0]`, not
///   `inputs[0].components[0].dim` (plain-struct fields get the full path);
/// - a whole-document type error (the root is not an object, e.g. `42`) has no
///   field path and still names the Rust struct (`expected struct ModelSpec`).
///   `json.dumps(dict)` never produces such input; only a raw native caller can.
fn de_spec<T: DeserializeOwned>(label: &str, json: &str) -> PyResult<T> {
    // Strip serde_json's "... at line N column M" tail: line/column is
    // meaningless to a caller who passed a dict, not a file. rsplit (not split):
    // the echoed content can itself contain " at line ", so drop only the
    // trailing locator, never a fragment of the real message.
    fn clean(raw: &str) -> &str {
        raw.rsplit_once(" at line ").map_or(raw, |(head, _)| head)
    }

    let mut deserializer = serde_json::Deserializer::from_str(json);
    let value = serde_path_to_error::deserialize::<_, T>(&mut deserializer).map_err(|error| {
        let raw = error.inner().to_string();
        let message = clean(&raw);
        // A whole-document parse failure (EOF / syntax of the root) resolves to
        // path "." -- no useful field path there. A value-level error (incl. a
        // non-finite overflow, which serde classifies as Syntax) keeps its path.
        let path = error.path().to_string();
        if path == "." {
            PyValueError::new_err(format!("invalid {label}: {message}"))
        } else {
            PyValueError::new_err(format!("invalid {label} at {path}: {message}"))
        }
    })?;
    // serde_path_to_error::deserialize stops after one value; unlike
    // serde_json::from_str it does NOT check for EOF. Without this, a valid
    // document followed by trailing junk would normalize green and malformed
    // wire input would look valid -- the regression this guards against.
    deserializer.end().map_err(|error| {
        PyValueError::new_err(format!("invalid {label}: {}", clean(&error.to_string())))
    })?;
    Ok(value)
}

/// Wire-vocabulary constants re-exported to Python. The `rlmesh-adapters`
/// crate is the single source of truth: bindings re-export, never re-declare.
const WIRE_CONSTANTS: &[(&str, &str)] = &[
    ("ENV_METADATA_KEY", rlmesh_adapters::v1::ENV_METADATA_KEY),
    (
        "MODEL_METADATA_KEY",
        rlmesh_adapters::v1::MODEL_METADATA_KEY,
    ),
    ("IMAGE_PRIMARY", roles::core::IMAGE_PRIMARY),
    ("IMAGE_SECONDARY", roles::core::IMAGE_SECONDARY),
    ("INSTRUCTION", roles::core::INSTRUCTION),
    ("JOINT_POS", roles::core::JOINT_POS),
    ("JOINT_VEL", roles::core::JOINT_VEL),
    ("IMAGE_WRIST", roles::manipulation::IMAGE_WRIST),
    ("EEF_POS", roles::manipulation::EEF_POS),
    ("EEF_ROT", roles::manipulation::EEF_ROT),
    ("GRIPPER_POS", roles::manipulation::GRIPPER_POS),
    ("EEF_POS_2", roles::manipulation::EEF_POS_2),
    ("EEF_ROT_2", roles::manipulation::EEF_ROT_2),
    ("GRIPPER_POS_2", roles::manipulation::GRIPPER_POS_2),
    ("ACTION_DELTA_POS", roles::manipulation::ACTION_DELTA_POS),
    ("ACTION_DELTA_ROT", roles::manipulation::ACTION_DELTA_ROT),
    ("ACTION_GRIPPER", roles::manipulation::ACTION_GRIPPER),
    (
        "ACTION_DELTA_POS_2",
        roles::manipulation::ACTION_DELTA_POS_2,
    ),
    (
        "ACTION_DELTA_ROT_2",
        roles::manipulation::ACTION_DELTA_ROT_2,
    ),
    ("ACTION_GRIPPER_2", roles::manipulation::ACTION_GRIPPER_2),
];

/// Stub-only declarations for the wire constants that [`register_constants`]
/// adds at runtime, so the generated `_rlmesh.pyi` (and downstream type
/// checkers) see them. `module_variable!` feeds the stub generator only and
/// is available solely under the `stub-gen` feature, so the whole block is
/// gated; the runtime registration below is the source of truth.
#[cfg(feature = "stub-gen")]
mod stub_constants {
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "ENV_METADATA_KEY", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "MODEL_METADATA_KEY", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "IMAGE_PRIMARY", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "IMAGE_SECONDARY", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "INSTRUCTION", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "JOINT_POS", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "JOINT_VEL", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "IMAGE_WRIST", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "EEF_POS", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "EEF_ROT", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "GRIPPER_POS", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "EEF_POS_2", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "EEF_ROT_2", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "GRIPPER_POS_2", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "ACTION_DELTA_POS", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "ACTION_DELTA_ROT", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "ACTION_GRIPPER", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "ACTION_DELTA_POS_2", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "ACTION_DELTA_ROT_2", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "ACTION_GRIPPER_2", String);
    pyo3_stub_gen::module_variable!(
        "rlmesh._rlmesh",
        "ROTATION_DIMS",
        std::collections::HashMap<String, u32>
    );
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "IMAGE_LAYOUTS", Vec<String>);
}

/// Register the wire-vocabulary constants on the `_rlmesh` module.
pub fn register_constants(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for (name, value) in WIRE_CONSTANTS {
        m.add(*name, *value)?;
    }
    let rotation_dims: BTreeMap<&str, u32> = rlmesh_adapters::v1::RotationEncoding::ALL
        .iter()
        .map(|encoding| (encoding.as_str(), encoding.dims()))
        .collect();
    m.add("ROTATION_DIMS", rotation_dims)?;
    let layouts: Vec<&str> = rlmesh_adapters::v1::ImageLayout::ALL
        .iter()
        .map(|layout| layout.as_str())
        .collect();
    m.add("IMAGE_LAYOUTS", layouts)?;
    Ok(())
}

pub(crate) fn decode_value(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    if let Some(tensor) = crate::spaces::extract_tensor(value)? {
        return Ok(Value::Tensor(tensor.inner.clone()));
    }
    if let Ok(text) = value.cast::<PyString>() {
        return Ok(Value::Text(text.extract()?));
    }
    if let Ok(bytes) = value.cast::<PyBytes>() {
        return Ok(Value::Bytes(bytes.as_bytes().to_vec()));
    }
    if value.cast::<PyBool>().is_ok() {
        return Ok(Value::Number(f64::from(value.extract::<bool>()?)));
    }
    if value.cast::<PyInt>().is_ok() || value.cast::<PyFloat>().is_ok() {
        return Ok(Value::Number(value.extract()?));
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        let mut out: BTreeMap<String, Value> = BTreeMap::new();
        for (key, item) in dict.iter() {
            out.insert(key.extract()?, decode_value(&item)?);
        }
        return Ok(Value::Map(out));
    }
    if let Ok(list) = value.cast::<PyList>() {
        let mut out = Vec::with_capacity(list.len());
        for item in list.iter() {
            out.push(decode_value(&item)?);
        }
        return Ok(Value::List(out));
    }
    if let Ok(tuple) = value.cast::<PyTuple>() {
        let mut out = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            out.push(decode_value(&item)?);
        }
        return Ok(Value::List(out));
    }
    Err(PyTypeError::new_err(
        "unsupported adapter value type; expected Tensor, str, bytes, number, list, tuple, or dict",
    ))
}

/// Decode only the top-level observation entries the adapter reads.
///
/// Decoding the whole observation would let an unused — possibly
/// unencodable — env key abort a step the model does not even depend on.
/// The top-level observation entry a (possibly dotted) plan key lives under.
/// The reserved `"."` denotes the flat/root observation and is its own
/// top-level key, not a dotted path to split.
fn top_level_key(key: &str) -> &str {
    if key == "." {
        return ".";
    }
    key.split('.').next().unwrap_or(key)
}

fn decode_referenced_obs(
    value: &Bound<'_, PyAny>,
    referenced: &BTreeSet<String>,
) -> PyResult<BTreeMap<String, Value>> {
    // Referenced keys may be dotted (nested Dict paths); decode the whole
    // top-level entry that contains each.
    let top_level: BTreeSet<&str> = referenced.iter().map(|key| top_level_key(key)).collect();
    let dict = value.cast::<PyDict>()?;
    let mut out: BTreeMap<String, Value> = BTreeMap::new();
    for (key, value) in dict.iter() {
        let key: String = key.extract()?;
        if top_level.contains(key.as_str()) {
            out.insert(key, decode_value(&value)?);
        }
    }
    Ok(out)
}

pub(crate) fn encode_value<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    match value {
        Value::Tensor(tensor) => crate::spaces::wrap_native_tensor(py, tensor.clone()),
        Value::Text(text) => Ok(text.into_pyobject(py)?.into_any()),
        Value::Bytes(bytes) => Ok(PyBytes::new(py, bytes).into_any()),
        Value::Number(number) => Ok(number.into_pyobject(py)?.into_any()),
        Value::List(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(encode_value(py, item)?)?;
            }
            Ok(list.into_any())
        }
        Value::Map(entries) => {
            let dict = PyDict::new(py);
            for (key, item) in entries {
                dict.set_item(key, encode_value(py, item)?)?;
            }
            Ok(dict.into_any())
        }
    }
}

/// Build a neutral Python dict from an adapter value map (`encode_value` each).
fn value_map_to_py<'py>(
    py: Python<'py>,
    values: &BTreeMap<String, Value>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (key, value) in values {
        dict.set_item(key, encode_value(py, value)?)?;
    }
    Ok(dict)
}

/// The custom-input hole, backed by host (Python) neutral callables.
///
/// Each entry is a `model_key -> callable(full_obs_neutral_dict) -> neutral_value`
/// the Python layer built (it does the framework-bridge + the user transform
/// internally). The engine calls [`CustomTransform::apply`] per declared custom
/// hole during `assemble_obs`; this materializes the full per-lane obs into
/// Python, runs the callable, and decodes the result back to a [`Value`].
pub(crate) struct PyCustomTransform {
    customs: HashMap<String, Py<PyAny>>,
}

impl PyCustomTransform {
    pub(crate) fn new(customs: HashMap<String, Py<PyAny>>) -> Self {
        Self { customs }
    }
}

impl CustomTransform for PyCustomTransform {
    fn apply(
        &self,
        model_key: &str,
        _entrypoint: &str,
        raw_obs: &BTreeMap<String, Value>,
    ) -> Result<Option<Value>, ApplyError> {
        let Some(callable) = self.customs.get(model_key) else {
            return Err(ApplyError::new(format!(
                "no custom transform registered for '{model_key}'"
            )));
        };
        Python::attach(|py| -> PyResult<Value> {
            let obs = value_map_to_py(py, raw_obs)?;
            let result = callable.call1(py, (obs,))?;
            decode_value(result.bind(py))
        })
        .map(Some)
        .map_err(|err| ApplyError::new(err.to_string()))
    }
}

/// The custom-encoding hole, backed by host (Python) neutral callables.
///
/// `obs` repacks enc-shimmed observation payload keys (before frame-stacking);
/// `action` repacks enc-shimmed action segments (before the native conversion).
/// Each is a neutral `callable(neutral) -> neutral` the Python layer built (it
/// does the numpy round-trip + width/dtype validation internally); `None` means
/// the route declares no encoding on that side.
pub(crate) struct PyEncodings {
    obs: Option<Py<PyAny>>,
    action: Option<Py<PyAny>>,
}

impl PyEncodings {
    pub(crate) fn new(obs: Option<Py<PyAny>>, action: Option<Py<PyAny>>) -> Self {
        Self { obs, action }
    }
}

impl EncodingTransform for PyEncodings {
    fn repack_obs(&self, payload: &mut BTreeMap<String, Value>) -> Result<(), ApplyError> {
        let Some(callable) = &self.obs else {
            return Ok(());
        };
        Python::attach(|py| -> PyResult<()> {
            let input = value_map_to_py(py, payload)?;
            let result = callable.call1(py, (input,))?;
            let dict = result.bind(py).cast::<PyDict>()?;
            payload.clear();
            for (key, value) in dict.iter() {
                payload.insert(key.extract()?, decode_value(&value)?);
            }
            Ok(())
        })
        .map_err(|err| ApplyError::new(err.to_string()))
    }

    fn repack_action(&self, action: &mut Value) -> Result<(), ApplyError> {
        let Some(callable) = &self.action else {
            return Ok(());
        };
        Python::attach(|py| -> PyResult<()> {
            let input = encode_value(py, action)?;
            let result = callable.call1(py, (input,))?;
            *action = decode_value(result.bind(py))?;
            Ok(())
        })
        .map_err(|err| ApplyError::new(err.to_string()))
    }
}

impl PyAdapterPlan {
    /// The native resolved adapter (for the served engine to drive directly).
    pub(crate) fn adapter(&self) -> &ResolvedAdapter {
        &self.adapter
    }
}

/// A resolved adapter plan handle backed by the `rlmesh-adapters` core.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(module = "rlmesh._rlmesh", name = "AdapterPlan", frozen)]
pub struct PyAdapterPlan {
    adapter: ResolvedAdapter,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl PyAdapterPlan {
    /// Human-readable summary of the resolved transformations.
    fn describe(&self) -> String {
        self.adapter.describe()
    }

    /// The top-level observation keys this adapter reads.
    ///
    /// A host wrapper should encode only these before calling
    /// [`transform_obs`](Self::transform_obs), so an unused — possibly
    /// unencodable — observation key never aborts a step.
    fn referenced_obs_keys(&self) -> Vec<String> {
        let mut keys: BTreeSet<String> = BTreeSet::new();
        for key in self.adapter.referenced_obs_keys() {
            keys.insert(top_level_key(&key).to_owned());
        }
        keys.into_iter().collect()
    }

    /// `(model_key, transform)` pairs for custom-input holes, plan order.
    fn custom_inputs(&self) -> Vec<(String, String)> {
        self.adapter
            .obs_plans
            .iter()
            .filter_map(|plan| match plan {
                ObsPlan::Custom(custom) => {
                    Some((custom.model_key.clone(), custom.transform.clone()))
                }
                _ => None,
            })
            .collect()
    }

    /// Apply the observation plans to a canonical value-tree observation map.
    ///
    /// Returns `{model_key: encoded_value}`; custom inputs are omitted
    /// (the caller fills them from the raw host observation).
    fn transform_obs<'py>(
        &self,
        py: Python<'py>,
        raw_obs: &Bound<'py, PyAny>,
    ) -> PyResult<BTreeMap<String, Py<PyAny>>> {
        let raw_obs = decode_referenced_obs(raw_obs, &self.adapter.referenced_obs_keys())?;
        let payload = self
            .adapter
            .transform_obs(&raw_obs, &SkipCustoms)
            .map_err(|err| PyValueError::new_err(err.message))?;
        let mut out: BTreeMap<String, Py<PyAny>> = BTreeMap::new();
        for (key, value) in &payload {
            out.insert(key.clone(), encode_value(py, value)?.unbind());
        }
        Ok(out)
    }

    /// Apply the action plan to a canonical value-tree model action.
    fn transform_action<'py>(
        &self,
        py: Python<'py>,
        raw_action: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let action = self
            .adapter
            .transform_action(&decode_value(raw_action)?)
            .map_err(|err| PyValueError::new_err(err.message))?;
        encode_value(py, &Value::Tensor(action))
    }
}

/// Resolve env tags + spaces and a model spec into a plan handle.
///
/// `observation_space`/`action_space` are gymnasium space objects; they are
/// parsed and projected into the adapters `SpaceView`. Custom-input
/// entrypoint trust is enforced by the Python wrapper *before* this call
/// (untrusted entrypoints never reach here), so the core resolves with
/// trust granted: every custom input that arrives is already vetted and is
/// kept as a host-filled hole.
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(
        module = "rlmesh._rlmesh",
        python = r#"
def adapters_resolve(env_tags_json: str, observation_space: object, action_space: object, model_spec_json: str) -> AdapterPlan: ...
"#
    )
)]
#[pyfunction]
pub fn adapters_resolve(
    env_tags_json: &str,
    observation_space: &Bound<'_, PyAny>,
    action_space: &Bound<'_, PyAny>,
    model_spec_json: &str,
) -> PyResult<PyAdapterPlan> {
    let tags: EnvTags = de_spec("env tags", env_tags_json)?;
    let model_spec: ModelSpec = de_spec("model spec", model_spec_json)?;
    let obs_view = SpaceView::from(&crate::spaces::parse_space(observation_space)?);
    let action_view = SpaceView::from(&crate::spaces::parse_space(action_space)?);
    let adapter = resolve(&tags, &obs_view, &action_view, &model_spec, true)
        .map_err(|err| PyValueError::new_err(err.message))?;
    Ok(PyAdapterPlan { adapter })
}

/// Validate env tags against the env's observation/action spaces.
///
/// Runs only the native `join` step that [`adapters_resolve`] performs
/// internally -- no model side. This surfaces tag/space mismatches at
/// authoring time (e.g. from `rlmesh.adapters.tag`) before any model is
/// paired against the env.
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(
        module = "rlmesh._rlmesh",
        python = r#"
def adapters_join_check(env_tags_json: str, observation_space: object, action_space: object) -> None: ...
"#
    )
)]
#[pyfunction]
pub fn adapters_join_check(
    env_tags_json: &str,
    observation_space: &Bound<'_, PyAny>,
    action_space: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let tags: EnvTags = de_spec("env tags", env_tags_json)?;
    let obs_view = SpaceView::from(&crate::spaces::parse_space(observation_space)?);
    let action_view = SpaceView::from(&crate::spaces::parse_space(action_space)?);
    join(&tags, &obs_view, &action_view).map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(())
}

/// Validate and canonicalize a spec's JSON through the Rust serde codec.
///
/// `side` is `"env"` (EnvTags) or `"model"` (ModelSpec). The JSON is parsed by
/// the authoritative serde codec -- enforcing the frozen vocabulary, field-name
/// strictness, dim bounds, the stack ceiling, ... -- and re-serialized to its
/// canonical form. This is the single normalize/validate door the Python codec
/// routes through, so the two languages cannot disagree on the format. When
/// `allow_custom` is false (the publish boundary) a custom input carrying an
/// entrypoint is rejected; resolve passes `allow_custom=true`.
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(
        module = "rlmesh._rlmesh",
        python = r#"
def adapters_spec_normalize(side: str, spec_json: str, allow_custom: bool) -> str: ...
"#
    )
)]
#[pyfunction]
pub fn adapters_spec_normalize(
    side: &str,
    spec_json: &str,
    allow_custom: bool,
) -> PyResult<String> {
    match side {
        "env" => {
            let tags: EnvTags = de_spec("env tags", spec_json)?;
            serde_json::to_string(&tags).map_err(|err| {
                PyValueError::new_err(format!("could not serialize env tags: {err}"))
            })
        }
        "model" => {
            let spec: ModelSpec = de_spec("model spec", spec_json)?;
            // Defense-in-depth at the publish boundary. Today the live gate is
            // Python's model_input_to_dict, which raises on any custom before a
            // spec ever reaches here, so every Python caller passes
            // allow_custom=true. This branch is the latent enforcement for raw
            // native callers and the future FE binding -- keep it as the codec's
            // own publish guard, not a redundancy.
            if !allow_custom {
                for input in &spec.inputs {
                    if let ModelInput::Custom(custom) = input {
                        return Err(PyValueError::new_err(format!(
                            "custom input {:?} carries an entrypoint ({:?}) and cannot be \
                             published in v1 contract metadata; resolve the spec locally",
                            custom.key, custom.transform
                        )));
                    }
                }
            }
            serde_json::to_string(&spec).map_err(|err| {
                PyValueError::new_err(format!("could not serialize model spec: {err}"))
            })
        }
        other => Err(PyValueError::new_err(format!(
            "unknown spec side {other:?}; expected \"env\" or \"model\""
        ))),
    }
}
