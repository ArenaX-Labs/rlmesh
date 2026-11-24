//! Bindings for the `rlmesh-adapters` core: spec resolution and plan application.
//!
//! The env side is sparse tags (JSON) over the env's gymnasium spaces; the model side is its
//! spec (JSON). [`adapters_resolve`] joins and resolves them into a plan handle. Values cross
//! as the canonical Python value tree; framework conversion stays in Python. Custom inputs are
//! never evaluated here: the plan keeps them as holes for the Python wrapper to fill.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Mutex;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use rlmesh_adapters::v1::{
    Advisory, ApplyError, CustomTransform, EncodingTransform, EnvTags, FrameBuffers, FramePolicy,
    InputNode, ModelLeaf, ModelSpec, NoEncodings, NodePath, ObsPlan, PathSeg, ResolvedAdapter,
    RolePolicy, SkipCustoms, SpaceView, SpaceViewKind, Value, assemble_obs,
    build_describe_envelope, join, observe_obs, record_action, reject_unframed_roles_env,
    reject_unframed_roles_model, reject_unknowns_env, reject_unknowns_model,
    reject_unsanctioned_roles_env, reject_unsanctioned_roles_model, resolve, roles,
};
use serde::de::DeserializeOwned;

const MAX_SPEC_BYTES: usize = 4 << 20;

fn de_spec<T: DeserializeOwned>(label: &str, json: &str) -> PyResult<T> {
    if json.len() > MAX_SPEC_BYTES {
        return Err(PyValueError::new_err(format!(
            "invalid {label}: spec is {} bytes, over the {MAX_SPEC_BYTES}-byte limit",
            json.len()
        )));
    }
    // rsplit, not split: the echoed content can itself contain " at line ".
    fn clean(raw: &str) -> &str {
        raw.rsplit_once(" at line ").map_or(raw, |(head, _)| head)
    }

    let mut deserializer = serde_json::Deserializer::from_str(json);
    let value = serde_path_to_error::deserialize::<_, T>(&mut deserializer).map_err(|error| {
        let raw = error.inner().to_string();
        let message = clean(&raw);
        let path = error.path().to_string();
        if path == "." {
            PyValueError::new_err(format!("invalid {label}: {message}"))
        } else {
            PyValueError::new_err(format!("invalid {label} at {path}: {message}"))
        }
    })?;
    // serde_path_to_error stops after one value and does not check for EOF: without this, a valid
    // document followed by trailing junk would pass.
    deserializer.end().map_err(|error| {
        PyValueError::new_err(format!("invalid {label}: {}", clean(&error.to_string())))
    })?;
    Ok(value)
}

const WIRE_CONSTANTS: &[(&str, &str)] = &[
    ("ENV_METADATA_KEY", rlmesh_adapters::v1::ENV_METADATA_KEY),
    (
        "ENV_BRANCH_METADATA_KEY",
        rlmesh_adapters::v1::ENV_BRANCH_METADATA_KEY,
    ),
    (
        "MODEL_METADATA_KEY",
        rlmesh_adapters::v1::MODEL_METADATA_KEY,
    ),
    ("ENV_RESET_OPTIONS_KEY", rlmesh::ENV_RESET_OPTIONS_KEY),
    (
        "ENV_RENDER_PARAMS_KEY",
        rlmesh::binding::runtime::ENV_RENDER_PARAMS_KEY,
    ),
    (
        "ENV_REQUIRES_EPISODE_CAP_KEY",
        rlmesh::binding::runtime::ENV_REQUIRES_EPISODE_CAP_KEY,
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
    ("EEF_WRENCH", roles::manipulation::EEF_WRENCH),
    ("ACTION_DELTA_POS", roles::manipulation::ACTION_DELTA_POS),
    ("ACTION_DELTA_ROT", roles::manipulation::ACTION_DELTA_ROT),
    ("ACTION_GRIPPER", roles::manipulation::ACTION_GRIPPER),
    ("ACTION_EEF_POS", roles::manipulation::ACTION_EEF_POS),
    ("ACTION_EEF_ROT", roles::manipulation::ACTION_EEF_ROT),
    ("BASE_ANG_VEL", roles::body::BASE_ANG_VEL),
    ("BASE_ROT", roles::body::BASE_ROT),
    ("COMMAND_BASE_VEL", roles::body::COMMAND_BASE_VEL),
    ("LEFT_ARM", roles::parts::LEFT_ARM),
    ("RIGHT_ARM", roles::parts::RIGHT_ARM),
    ("HEAD", roles::parts::HEAD),
    ("TORSO", roles::parts::TORSO),
    ("BASE", roles::parts::BASE),
    ("LEFT_LEG", roles::parts::LEFT_LEG),
    ("RIGHT_LEG", roles::parts::RIGHT_LEG),
];

#[cfg(feature = "stub-gen")]
mod stub_constants {
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "ENV_METADATA_KEY", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "ENV_BRANCH_METADATA_KEY", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "MODEL_METADATA_KEY", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "ENV_RESET_OPTIONS_KEY", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "ENV_RENDER_PARAMS_KEY", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "ENV_REQUIRES_EPISODE_CAP_KEY", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "DESCRIBE_METADATA_KEY", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "DESCRIBE_SCHEMA_VERSION", u32);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "SESSION_WIRE_REVISION", u32);
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
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "EEF_WRENCH", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "ACTION_DELTA_POS", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "ACTION_DELTA_ROT", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "ACTION_GRIPPER", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "ACTION_EEF_POS", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "ACTION_EEF_ROT", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "BASE_ANG_VEL", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "BASE_ROT", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "COMMAND_BASE_VEL", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "LEFT_ARM", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "RIGHT_ARM", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "HEAD", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "TORSO", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "BASE", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "LEFT_LEG", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "RIGHT_LEG", String);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "PARTS", Vec<String>);
    pyo3_stub_gen::module_variable!(
        "rlmesh._rlmesh",
        "ROTATION_DIMS",
        std::collections::HashMap<String, u32>
    );
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "IMAGE_LAYOUTS", Vec<String>);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "RESAMPLES", Vec<String>);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "CROP_MODES", Vec<String>);
    pyo3_stub_gen::module_variable!("rlmesh._rlmesh", "CHANNEL_ORDERS", Vec<String>);
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
    m.add("PARTS", roles::parts::PARTS.to_vec())?;
    m.add("RESAMPLES", rlmesh_adapters::v1::RESAMPLES.to_vec())?;
    m.add("CROP_MODES", rlmesh_adapters::v1::CROP_MODES.to_vec())?;
    m.add(
        "CHANNEL_ORDERS",
        rlmesh_adapters::v1::CHANNEL_ORDERS.to_vec(),
    )?;
    m.add(
        "DESCRIBE_SCHEMA_VERSION",
        rlmesh_adapters::v1::DESCRIBE_SCHEMA_VERSION,
    )?;
    m.add(
        "SESSION_WIRE_REVISION",
        rlmesh_session::wire::version::WIRE_REVISION,
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
        Value::None => Ok(py.None().into_bound(py)),
        Value::Bool(value) => Ok(value.into_pyobject(py)?.to_owned().into_any()),
        Value::Int(value) => Ok(value.into_pyobject(py)?.into_any()),
        Value::Tuple(items) => Ok(PyTuple::new(
            py,
            items
                .iter()
                .map(|v| encode_value(py, v))
                .collect::<PyResult<Vec<_>>>()?,
        )?
        .into_any()),
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

/// One non-fatal adapter advisory: a message plus a severity tier.
///
/// `severity` is `"info"` for a benign hint and `"caution"` when the adapter substituted or
/// fabricated model-visible data (a role-rebound camera, a zero-filled frame).
/// `str(advisory)` is the message.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(module = "rlmesh._rlmesh", name = "Advisory", frozen, eq, hash)]
#[derive(PartialEq, Eq, Hash)]
pub struct PyAdvisory {
    advisory: Advisory,
}

#[cfg_attr(feature = "stub-gen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "stub-gen"), pyo3_stub_gen_derive::remove_gen_stub)]
#[pymethods]
impl PyAdvisory {
    /// The severity tier: `"info"` or `"caution"`.
    #[getter]
    #[gen_stub(override_return_type(type_repr = "typing.Literal[\"info\", \"caution\"]", imports = ("typing")))]
    fn severity(&self) -> &'static str {
        self.advisory.severity.as_str()
    }

    /// The human-readable note.
    #[getter]
    fn message(&self) -> &str {
        &self.advisory.message
    }

    fn __str__(&self) -> &str {
        &self.advisory.message
    }

    fn __repr__(&self) -> String {
        format!(
            "Advisory(severity={:?}, message={:?})",
            self.advisory.severity.as_str(),
            self.advisory.message
        )
    }
}

impl From<Advisory> for PyAdvisory {
    fn from(advisory: Advisory) -> Self {
        Self { advisory }
    }
}

type StateLayout = (Py<PyList>, Vec<u32>, u32);

const LOCAL_EPISODE: &str = "";

/// A resolved adapter plan handle backed by the `rlmesh-adapters` core.
#[cfg_attr(feature = "stub-gen", gen_stub_pyclass)]
#[pyclass(module = "rlmesh._rlmesh", name = "AdapterPlan", frozen)]
pub struct PyAdapterPlan {
    adapter: ResolvedAdapter,
    /// In-process frame windows for inputs that declare a history; empty when none is stacked.
    windows: Mutex<FrameBuffers>,
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
    /// Each note carries a severity tier (`"info"` / `"caution"`).
    fn advisories(&self) -> Vec<PyAdvisory> {
        rlmesh_adapters::v1::advisories(&self.adapter)
            .into_iter()
            .map(PyAdvisory::from)
            .collect()
    }

    /// The top-level observation keys this adapter reads. Encode only these before
    /// `transform_obs`, so an unused, possibly unencodable key never aborts a step.
    fn referenced_obs_keys(&self) -> Vec<String> {
        self.adapter
            .referenced_obs_keys()
            .into_iter()
            .collect::<BTreeSet<String>>()
            .into_iter()
            .collect()
    }

    /// `(segments, transform)` pairs for custom-input holes, plan order.
    ///
    /// `segments` is the leaf's placement path: each step a `str` (`Dict` key) or `int`
    /// (`Tuple` index); an empty list is the root.
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

    /// `(segments, part_widths, width)` per resolved state input, plan order.
    ///
    /// `segments` is the leaf's placement path (as in `custom_inputs`), `part_widths` the
    /// width each declared part contributes in order, and `width` the assembled leaf's width
    /// (its `pad_to` when it pads). A part's offset is the sum of the widths before it. A
    /// state whose widths are not all statically known is omitted.
    #[gen_stub(override_return_type(
        type_repr = "builtins.list[tuple[builtins.list[builtins.str | builtins.int], \
                     builtins.list[builtins.int], builtins.int]]",
        imports = ("builtins")
    ))]
    fn state_layouts<'py>(&self, py: Python<'py>) -> PyResult<Vec<StateLayout>> {
        self.adapter
            .obs_plans
            .iter()
            .filter_map(|plan| match plan {
                ObsPlan::State(state) => Some(state),
                _ => None,
            })
            .filter_map(|state| {
                let native_width = state.native_width?;
                let widths: Vec<u32> = state
                    .pieces
                    .iter()
                    .map(|piece| piece.width)
                    .collect::<Option<_>>()?;
                Some((
                    &state.placement,
                    widths,
                    state.pad_to.unwrap_or(native_width),
                ))
            })
            .map(|(placement, widths, width)| {
                Ok((path_segments_to_py(py, placement)?, widths, width))
            })
            .collect()
    }

    /// Apply the observation plans to a canonical value-tree observation map.
    ///
    /// Returns the assembled payload as a nested dict/list/leaf value tree in the model
    /// spec's input shape. Custom inputs are omitted; the caller fills them from the raw
    /// observation.
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing")))]
    fn transform_obs<'py>(
        &self,
        py: Python<'py>,
        raw_obs: &Bound<'py, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let raw_obs = decode_referenced_obs(raw_obs, &self.adapter.referenced_obs_keys())?;
        let mut windows = self.windows.lock().expect("frame windows");
        let step = self.tick(&mut windows)?;
        let payload = assemble_obs(
            &self.adapter,
            &raw_obs,
            LOCAL_EPISODE,
            step,
            &mut windows,
            &SkipCustoms,
            &NoEncodings,
        )
        .map_err(|err| PyValueError::new_err(err.message))?;
        Ok(encode_value(py, &payload)?.unbind())
    }

    /// Canonical placement strings of the inputs that hold per-episode history:
    /// a frame window, or a part reading the model's previous action. Empty means a step that
    /// predicts nothing has no state to advance, so `transform_history` can be skipped.
    fn history_keys(&self) -> Vec<String> {
        self.adapter.history_keys()
    }

    /// `(key, span, frame_bytes)` per frame window held per live episode; `frame_bytes` is `0` when
    /// the env camera resolution was not derivable.
    fn history_windows(&self) -> Vec<(String, u32, u64)> {
        self.adapter
            .history_windows()
            .into_iter()
            .map(|window| (window.key, window.span, window.frame_bytes))
            .collect()
    }

    /// Advance the frame windows from a raw observation, assembling nothing. Call it for a
    /// step that replays a queued action: the frame still goes in the window, and the local
    /// step counter advances.
    fn transform_history<'py>(&self, raw_obs: &Bound<'py, PyAny>) -> PyResult<()> {
        let raw_obs = decode_referenced_obs(raw_obs, &self.adapter.referenced_obs_keys())?;
        let mut windows = self.windows.lock().expect("frame windows");
        self.tick(&mut windows)?;
        observe_obs(&self.adapter, &raw_obs, LOCAL_EPISODE, &mut windows)
            .map_err(|err| PyValueError::new_err(err.message))
    }

    /// Drop the in-process frame windows (and the local step counter) at an
    /// episode boundary.
    fn reset_history(&self) {
        self.windows.lock().expect("frame windows").clear();
    }

    /// Apply the action plan to a canonical value-tree model action, recording it as the
    /// action executed at the last ticked step when a state part reads the previous action.
    fn transform_action<'py>(
        &self,
        py: Python<'py>,
        raw_action: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let raw_action = decode_value(raw_action)?;
        let action = rlmesh_adapters::v1::transform_action(&self.adapter, &raw_action)
            .map_err(|err| PyValueError::new_err(err.message))?;
        if self.adapter.reads_previous_action() {
            let mut windows = self.windows.lock().expect("frame windows");
            if let Some(step) = windows.last_step(LOCAL_EPISODE) {
                record_action(
                    &self.adapter,
                    &raw_action,
                    &mut windows,
                    LOCAL_EPISODE,
                    step,
                )
                .map_err(|err| PyValueError::new_err(err.message))?;
            }
        }
        encode_value(py, &Value::Tensor(action))
    }
}

impl PyAdapterPlan {
    fn tick(&self, windows: &mut FrameBuffers) -> PyResult<i64> {
        let step = windows.next_step(LOCAL_EPISODE);
        if self.adapter.holds_history() {
            windows
                .advance_step(LOCAL_EPISODE, step)
                .map_err(|err| PyValueError::new_err(err.message))?;
        }
        Ok(step)
    }
}

/// Resolve env tags + spaces and a model spec into a plan handle.
///
/// `observation_space` and `action_space` are gymnasium space objects. Custom inputs resolve
/// as trusted host-filled holes, so the caller must vet their entrypoints first.
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
    Ok(PyAdapterPlan {
        adapter,
        windows: Mutex::new(FrameBuffers::new()),
    })
}

/// Validate env tags against the env's observation/action spaces.
///
/// Runs only the env side of `adapters_resolve`, so tag/space mismatches surface before any
/// model is paired. A space may also be given as its describe-envelope dict. Returns the
/// advisories.
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(
        module = "rlmesh._rlmesh",
        python = r#"
def adapters_join_check(env_tags_json: str, observation_space: object, action_space: object) -> list[Advisory]: ...
"#
    )
)]
#[pyfunction]
pub fn adapters_join_check(
    env_tags_json: &str,
    observation_space: &Bound<'_, PyAny>,
    action_space: &Bound<'_, PyAny>,
) -> PyResult<Vec<PyAdvisory>> {
    let tags: EnvTags = de_spec("env tags", env_tags_json)?;
    reject_unknowns_env(&tags)
        .map_err(|message| PyValueError::new_err(format!("invalid env tags: {message}")))?;
    let obs_view = join_space_view(observation_space)?;
    let action_view = join_space_view(action_space)?;
    let features = join(&tags, &obs_view, &action_view)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(features
        .advisories
        .into_iter()
        .map(PyAdvisory::from)
        .collect())
}

fn join_space_view(space: &Bound<'_, PyAny>) -> PyResult<SpaceView> {
    match space.cast::<PyDict>() {
        Ok(envelope) => envelope_space_view(envelope),
        Err(_) => Ok(SpaceView::from(&crate::spaces::parse_space(space)?)),
    }
}

fn envelope_space_view(space: &Bound<'_, PyDict>) -> PyResult<SpaceView> {
    let required = |key: &str| {
        space
            .get_item(key)?
            .ok_or_else(|| PyValueError::new_err(format!("envelope space is missing {key:?}")))
    };
    let kind_name: String = required("kind")?.extract()?;
    let kind = match kind_name.as_str() {
        "box" => SpaceViewKind::Box,
        "discrete" => SpaceViewKind::Discrete,
        "multi_binary" => SpaceViewKind::MultiBinary,
        "multi_discrete" => SpaceViewKind::MultiDiscrete,
        "text" => SpaceViewKind::Text,
        "dict" => SpaceViewKind::Dict,
        "tuple" => SpaceViewKind::Tuple,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown envelope space kind {other:?}"
            )));
        }
    };
    let details = match space.get_item("details")? {
        Some(details) if !details.is_none() => details.cast_into::<PyDict>()?,
        _ => PyDict::new(space.py()),
    };
    let bound = |key: &str, infinity: f64| -> PyResult<Option<Vec<f64>>> {
        match details.get_item(key)? {
            Some(value) if kind == SpaceViewKind::Box => {
                if value.is_none() {
                    return Ok(Some(vec![infinity]));
                }
                match value.extract::<f64>() {
                    Ok(uniform) => Ok(Some(vec![uniform])),
                    Err(_) => Ok(Some(
                        value
                            .extract::<Vec<Option<f64>>>()?
                            .into_iter()
                            .map(|edge| edge.unwrap_or(infinity))
                            .collect(),
                    )),
                }
            }
            _ => Ok(None),
        }
    };
    let (mut keys, mut children) = (Vec::new(), Vec::new());
    if let Some(spaces) = details.get_item("spaces")? {
        if let Ok(entries) = spaces.cast::<PyDict>() {
            for (key, child) in entries.iter() {
                keys.push(key.extract()?);
                children.push(envelope_space_view(&child.cast_into()?)?);
            }
        } else {
            for child in spaces.try_iter()? {
                children.push(envelope_space_view(&child?.cast_into()?)?);
            }
        }
    }
    Ok(SpaceView {
        kind,
        shape: required("shape")?.extract()?,
        dtype: required("dtype")?.extract()?,
        low: bound("low", f64::NEG_INFINITY)?,
        high: bound("high", f64::INFINITY)?,
        keys,
        children,
    })
}

/// Validate and canonicalize a spec's JSON through the Rust serde codec.
///
/// `side` is `"env"` (EnvTags) or `"model"` (ModelSpec). Returns the canonical JSON. With
/// `allow_custom` false (the publish boundary), a custom input carrying an entrypoint is
/// rejected.
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(
        module = "rlmesh._rlmesh",
        python = r#"
def adapters_spec_normalize(side: str, spec_json: str, allow_custom: bool, role_policy: str = "passthrough", require_frames: bool = False) -> str: ...
"#
    )
)]
#[pyfunction]
#[pyo3(signature = (side, spec_json, allow_custom, role_policy = "passthrough", require_frames = false))]
pub fn adapters_spec_normalize(
    side: &str,
    spec_json: &str,
    allow_custom: bool,
    role_policy: &str,
    require_frames: bool,
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
    let frame_gate = if require_frames {
        FramePolicy::Require
    } else {
        FramePolicy::Allow
    };
    match side {
        "env" => {
            let tags: EnvTags = de_spec("env tags", spec_json)?;
            reject_unknowns_env(&tags)
                .map_err(|message| PyValueError::new_err(format!("invalid env tags: {message}")))?;
            if let Some(policy) = role_gate {
                reject_unsanctioned_roles_env(&tags, policy).map_err(|message| {
                    PyValueError::new_err(format!("invalid env tags: {message}"))
                })?;
            }
            reject_unframed_roles_env(&tags, frame_gate)
                .map_err(|message| PyValueError::new_err(format!("invalid env tags: {message}")))?;
            rlmesh::binding::adapter_json_format(
                &serde_json::to_value(&tags)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?,
            )
            .map_err(|err| PyValueError::new_err(format!("could not serialize env tags: {err}")))
        }
        "model" => {
            let spec: ModelSpec = de_spec("model spec", spec_json)?;
            reject_unknowns_model(&spec).map_err(|message| {
                PyValueError::new_err(format!("invalid model spec: {message}"))
            })?;
            if let Some(policy) = role_gate {
                reject_unsanctioned_roles_model(&spec, policy).map_err(|message| {
                    PyValueError::new_err(format!("invalid model spec: {message}"))
                })?;
            }
            reject_unframed_roles_model(&spec, frame_gate).map_err(|message| {
                PyValueError::new_err(format!("invalid model spec: {message}"))
            })?;
            if !allow_custom {
                reject_custom_leaves(&spec.input)?;
            }
            rlmesh::binding::adapter_json_format(
                &serde_json::to_value(&spec)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?,
            )
            .map_err(|err| PyValueError::new_err(format!("could not serialize model spec: {err}")))
        }
        other => Err(PyValueError::new_err(format!(
            "unknown spec side {other:?}; expected \"env\" or \"model\""
        ))),
    }
}

/// Build the canonical describe-envelope JSON string from gathered pieces.
///
/// `kind` is `"env"` or `"model"`; `pieces_json` is one JSON object of the gathered pieces.
/// `generated_at`, if given, must be RFC 3339; none is stamped otherwise. The bytes are
/// identical across producer languages, so persist the string verbatim.
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

/// The one camera resolution a model spec asserts, as `(height, width)`, or
/// `None` when no image input declares a `render`. Two inputs asserting different sizes
/// raise `ValueError`.
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(
        module = "rlmesh._rlmesh",
        python = r#"
def adapters_render_size(spec_json: str) -> tuple[int, int] | None: ...
"#
    )
)]
#[pyfunction]
pub fn adapters_render_size(spec_json: &str) -> PyResult<Option<(u32, u32)>> {
    let spec: ModelSpec = de_spec("model spec", spec_json)?;
    let requests = rlmesh_adapters::v1::render_requests(&spec);
    let Some((_, size)) = requests.first() else {
        return Ok(None);
    };
    if requests.iter().any(|(_, other)| other != size) {
        let detail = requests
            .iter()
            .map(|(role, (h, w))| format!("{role} -> {h}x{w}"))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(PyValueError::new_err(format!(
            "the model's image inputs assert different render sizes for one camera dial: {detail}"
        )));
    }
    Ok(Some(*size))
}
