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
    ApplyError, CustomTransform, EncodingTransform, EnvTags, InputNode, ModelLeaf, ModelSpec,
    NodePath, ObsPlan, PathSeg, ResolvedAdapter, RolePolicy, SkipCustoms, SpaceView, Value,
    build_describe_envelope, join, reject_unknowns_env, reject_unknowns_model,
    reject_unsanctioned_roles_env, reject_unsanctioned_roles_model, resolve, roles,
};
use serde::de::DeserializeOwned;

/// Total-byte ceiling on an untrusted spec blob. A spec now originates from a
/// possibly-untrusted env contract and is retained + re-emitted (the tolerant
/// reader keeps unknown kinds/fields verbatim), so an unbounded blob is a DoS
/// amplifier. 4 MiB is absurdly generous for any real spec yet caps a hostile
/// contract; JSON nesting depth is separately bounded by serde_json's default
/// 128-deep recursion limit, and per-dimension size by `num::MAX_DIM`.
const MAX_SPEC_BYTES: usize = 4 << 20;

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
    if json.len() > MAX_SPEC_BYTES {
        return Err(PyValueError::new_err(format!(
            "invalid {label}: spec is {} bytes, over the {MAX_SPEC_BYTES}-byte limit",
            json.len()
        )));
    }
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
    (
        "DESCRIBE_METADATA_KEY",
        rlmesh_adapters::v1::DESCRIBE_METADATA_KEY,
    ),
    ("IMAGE_PRIMARY", roles::core::IMAGE_PRIMARY),
    ("IMAGE_SECONDARY", roles::core::IMAGE_SECONDARY),
    ("INSTRUCTION", roles::core::INSTRUCTION),
    ("JOINT_POS", roles::core::JOINT_POS),
    ("JOINT_VEL", roles::core::JOINT_VEL),
    ("ACTION_JOINT_POS", roles::core::ACTION_JOINT_POS),
    ("ACTION_JOINT_VEL", roles::core::ACTION_JOINT_VEL),
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
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "DESCRIBE_METADATA_KEY", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "DESCRIBE_SCHEMA_VERSION", u32);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "IMAGE_PRIMARY", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "IMAGE_SECONDARY", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "INSTRUCTION", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "JOINT_POS", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "JOINT_VEL", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "ACTION_JOINT_POS", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "ACTION_JOINT_VEL", String);
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
    // The describe-envelope schema version is a u32, not a string, so it can't
    // ride WIRE_CONSTANTS; add it directly. Rust is the sole writer of this.
    m.add(
        "DESCRIBE_SCHEMA_VERSION",
        rlmesh_adapters::v1::DESCRIBE_SCHEMA_VERSION,
    )?;
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
///
/// `referenced` carries raw-obs *envelope keys* (already top-level: the first
/// path segment of each Dict-rooted source, or the reserved root key for a
/// root/Tuple-rooted source — see `rlmesh_adapters::v1` `referenced_obs_keys`),
/// so no dotted splitting is needed here.
fn decode_referenced_obs(
    value: &Bound<'_, PyAny>,
    referenced: &BTreeSet<String>,
) -> PyResult<BTreeMap<String, Value>> {
    let dict = value.cast::<PyDict>()?;
    let mut out: BTreeMap<String, Value> = BTreeMap::new();
    for (key, value) in dict.iter() {
        let key: String = key.extract()?;
        if referenced.contains(key.as_str()) {
            out.insert(key, decode_value(&value)?);
        }
    }
    Ok(out)
}

/// Reject any `Custom` leaf in a model input tree, the publish-boundary guard
/// for raw native / future-FE callers (the Python `model_input_to_dict` is the
/// live gate). Walks the recursive tree to every leaf.
fn reject_custom_leaves(node: &InputNode) -> PyResult<()> {
    match node {
        InputNode::Leaf(ModelLeaf::Custom(custom)) => Err(PyValueError::new_err(format!(
            "custom input carries an entrypoint ({:?}) and cannot be published in v1 contract \
             metadata; resolve the spec locally",
            custom.transform
        ))),
        InputNode::Leaf(_) => Ok(()),
        InputNode::Dict(map) => map.values().try_for_each(reject_custom_leaves),
        InputNode::Tuple(items) => items.iter().try_for_each(reject_custom_leaves),
    }
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

/// Build a Python list of placement segments from a native [`NodePath`].
///
/// Each step is a `str` (`Dict` key) or `int` (`Tuple` index); an empty list is
/// the root. Hands the caller the structured path directly, so nothing has to
/// re-parse a rendered placement string.
fn path_segments_to_py<'py>(py: Python<'py>, path: &NodePath) -> PyResult<Py<PyList>> {
    let segments = PyList::empty(py);
    for segment in &path.0 {
        match segment {
            PathSeg::Key(key) => segments.append(key)?,
            PathSeg::Index(index) => segments.append(*index)?,
        }
    }
    Ok(segments.unbind())
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
    fn repack_obs(&self, payload: &mut Value) -> Result<(), ApplyError> {
        let Some(callable) = &self.obs else {
            return Ok(());
        };
        Python::attach(|py| -> PyResult<()> {
            // The payload is now a Value tree; hand the encoded tree to the
            // Python shim and decode its result back in place.
            let input = encode_value(py, payload)?;
            let result = callable.call1(py, (input,))?;
            *payload = decode_value(result.bind(py))?;
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

    /// Per-env data-loss / fabrication notes (zero-filled camera, aspect crop):
    /// the "warn" subset of `describe`, empty when nothing noteworthy happened.
    fn advisories(&self) -> Vec<String> {
        self.adapter.advisories()
    }

    /// The top-level observation keys this adapter reads.
    ///
    /// A host wrapper should encode only these before calling
    /// [`transform_obs`](Self::transform_obs), so an unused — possibly
    /// unencodable — observation key never aborts a step.
    fn referenced_obs_keys(&self) -> Vec<String> {
        // The adapter already reports raw-obs envelope keys (top-level by
        // construction); surface them directly.
        self.adapter
            .referenced_obs_keys()
            .into_iter()
            .collect::<BTreeSet<String>>()
            .into_iter()
            .collect()
    }

    /// `(segments, transform)` pairs for custom-input holes, plan order.
    ///
    /// `segments` is the custom leaf's structured placement path: each step is a
    /// `str` (`Dict` key) or `int` (`Tuple` index), built directly from the
    /// native `NodePath` — no placement string for the caller to re-parse. An
    /// empty segment list is the root (a bare-leaf payload).
    #[gen_stub(override_return_type(
        type_repr = "builtins.list[tuple[builtins.list[builtins.str | builtins.int], builtins.str]]",
        imports = ("builtins")
    ))]
    fn custom_inputs<'py>(&self, py: Python<'py>) -> PyResult<Vec<(Py<PyList>, String)>> {
        self.adapter
            .obs_plans
            .iter()
            .filter_map(|plan| match plan {
                ObsPlan::Custom(custom) => Some((&custom.placement, &custom.transform)),
                _ => None,
            })
            .map(|(placement, transform)| {
                Ok((path_segments_to_py(py, placement)?, transform.clone()))
            })
            .collect()
    }

    /// Apply the observation plans to a canonical value-tree observation map.
    ///
    /// Returns the assembled payload as a neutral Python object (a nested
    /// dict/list/leaf — the model spec's `InputNode` shape, a Value tree);
    /// custom inputs are omitted (the caller fills them from the raw host
    /// observation).
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing")))]
    fn transform_obs<'py>(
        &self,
        py: Python<'py>,
        raw_obs: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let raw_obs = decode_referenced_obs(raw_obs, &self.adapter.referenced_obs_keys())?;
        let payload = self
            .adapter
            .transform_obs(&raw_obs, &SkipCustoms)
            .map_err(|err| PyValueError::new_err(err.message))?;
        Ok(encode_value(py, &payload)?.unbind())
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
def adapters_join_check(env_tags_json: str, observation_space: object, action_space: object) -> list[str]: ...
"#
    )
)]
#[pyfunction]
pub fn adapters_join_check(
    env_tags_json: &str,
    observation_space: &Bound<'_, PyAny>,
    action_space: &Bound<'_, PyAny>,
) -> PyResult<Vec<String>> {
    let tags: EnvTags = de_spec("env tags", env_tags_json)?;
    // Authoring-time check is a PUBLISH door: an env validating its own tags must
    // see a typo or unbuildable kind now, not relay it to a peer.
    reject_unknowns_env(&tags)
        .map_err(|message| PyValueError::new_err(format!("invalid env tags: {message}")))?;
    let obs_view = SpaceView::from(&crate::spaces::parse_space(observation_space)?);
    let action_view = SpaceView::from(&crate::spaces::parse_space(action_space)?);
    // Hard tag/space disagreements still raise; non-fatal hints (e.g. a layout
    // that looks mis-declared) come back so the author sees them at `tag()` time.
    let features = join(&tags, &obs_view, &action_view)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(features.advisories)
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
def adapters_spec_normalize(side: str, spec_json: str, allow_custom: bool, role_policy: str = "passthrough") -> str: ...
"#
    )
)]
#[pyfunction]
#[pyo3(signature = (side, spec_json, allow_custom, role_policy = "passthrough"))]
pub fn adapters_spec_normalize(
    side: &str,
    spec_json: &str,
    allow_custom: bool,
    role_policy: &str,
) -> PyResult<String> {
    let role_gate = match role_policy {
        "passthrough" => None,
        "strict" => Some(RolePolicy::Strict),
        "forbid" => Some(RolePolicy::Forbid),
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown role_policy {other:?}; expected \"passthrough\", \"strict\", or \"forbid\""
            )));
        }
    };
    match side {
        "env" => {
            let tags: EnvTags = de_spec("env tags", spec_json)?;
            // PUBLISH mode: the strict-v1 gate. A typo'd field or an
            // unrecognized leaf kind dies here at the trust boundary, never
            // reaching a peer. The READ door (`adapters_resolve`) skips this.
            reject_unknowns_env(&tags)
                .map_err(|message| PyValueError::new_err(format!("invalid env tags: {message}")))?;
            if let Some(policy) = role_gate {
                reject_unsanctioned_roles_env(&tags, policy).map_err(|message| {
                    PyValueError::new_err(format!("invalid env tags: {message}"))
                })?;
            }
            serde_json::to_string(&tags).map_err(|err| {
                PyValueError::new_err(format!("could not serialize env tags: {err}"))
            })
        }
        "model" => {
            let spec: ModelSpec = de_spec("model spec", spec_json)?;
            // PUBLISH mode: the strict-v1 gate (see the env branch).
            reject_unknowns_model(&spec).map_err(|message| {
                PyValueError::new_err(format!("invalid model spec: {message}"))
            })?;
            if let Some(policy) = role_gate {
                reject_unsanctioned_roles_model(&spec, policy).map_err(|message| {
                    PyValueError::new_err(format!("invalid model spec: {message}"))
                })?;
            }
            // Defense-in-depth at the publish boundary. Today the live gate is
            // Python's model_input_to_dict, which raises on any custom before a
            // spec ever reaches here, so every Python caller passes
            // allow_custom=true. This branch is the latent enforcement for raw
            // native callers and the future FE binding -- keep it as the codec's
            // own publish guard, not a redundancy.
            if !allow_custom {
                reject_custom_leaves(&spec.input)?;
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

/// Build the canonical describe-envelope JSON string from gathered pieces.
///
/// `kind` is `"env"` or `"model"`. `pieces_json` is the producer-gathered
/// sub-pieces (target/env_spec/env_tags/model_spec/params/variants/runtime) as
/// one JSON object. `generated_at`, if given, must be RFC-3339 (caller-supplied,
/// never wall-clock here -- a build pins a reproducible value or omits it). The
/// Rust core stamps `schema_version`, enforces the env/model invariant, and
/// re-serializes the whole tree so the bytes are identical across producer
/// languages. The returned string should be kept verbatim (no `json.loads`
/// round-trip) when persisting to OCI metadata.
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(
        module = "rlmesh._rlmesh",
        python = r#"
def describe_envelope_normalize(kind: str, pieces_json: str, generated_at: str | None = None) -> str: ...
"#
    )
)]
#[pyfunction]
#[pyo3(signature = (kind, pieces_json, generated_at=None))]
pub fn describe_envelope_normalize(
    kind: &str,
    pieces_json: &str,
    generated_at: Option<&str>,
) -> PyResult<String> {
    build_describe_envelope(kind, pieces_json, generated_at)
        .map_err(|err| PyValueError::new_err(format!("invalid describe envelope: {err}")))
}
