//! Bindings for the `rlmesh-adapters` core: spec resolution and plan
//! application.
//!
//! The env side is given as sparse annotations (JSON) over the env's
//! observation/action spaces (gymnasium space objects, parsed here into the
//! adapters [`SpaceView`]); the model side is its declared spec (JSON).
//! [`adapters_resolve`] joins and resolves them into a plan handle.
//!
//! Values cross the boundary in a small tagged-tuple encoding produced by
//! `rlmesh.adapters.helpers.bridge`:
//!
//! - `("a", dtype, shape, bytes)` — a dense array (little-endian element
//!   bytes, matching the repo-wide tensor/scalar codec)
//! - `("b", bytes)` — an encoded image (PNG/JPEG), decoded here to an RGB
//!   uint8 HWC array (codec-level bridge behavior, not part of the pinned v1
//!   semantics)
//! - `("t", str)` — text
//! - `("n", float)` — a scalar number
//! - `("l", [encoded, ...])` — a list
//! - `("m", {key: encoded})` — a nested mapping
//!
//! Custom inputs are never evaluated here: the plan keeps them as holes
//! ([`SkipCustoms`]) and the Python wrapper runs the user's callable on the
//! raw Python observation afterwards.

use std::collections::{BTreeMap, BTreeSet};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
#[cfg(feature = "stub-gen")]
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use rlmesh_adapters::v1::{
    EnvAnnotations, ModelSpec, ObsPlan, ResolvedAdapter, SkipCustoms, SpaceView, Value, resolve,
    roles,
};
use rlmesh_spaces::{DType, Tensor};

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

fn decode_value(encoded: &Bound<'_, PyAny>) -> PyResult<Value> {
    let tuple = encoded.cast::<PyTuple>()?;
    let tag: String = tuple.get_item(0)?.extract()?;
    match tag.as_str() {
        "a" => {
            let dtype = DType::from_name(&tuple.get_item(1)?.extract::<String>()?)
                .ok_or_else(|| PyValueError::new_err("unsupported array dtype".to_owned()))?;
            let shape: Vec<i64> = tuple.get_item(2)?.extract()?;
            let raw: Vec<u8> = tuple.get_item(3)?.extract()?;
            // `Tensor::from_vec` validates that the byte length matches the
            // shape and dtype, so a short or mismatched buffer is a clean
            // error rather than a panic in the chunker.
            let tensor = Tensor::from_vec(raw, shape, dtype)
                .map_err(|err| PyValueError::new_err(format!("invalid array value: {err}")))?;
            Ok(Value::Tensor(tensor))
        }
        "b" => {
            let raw: Vec<u8> = tuple.get_item(1)?.extract()?;
            let decoded = image::load_from_memory(&raw)
                .map_err(|err| {
                    PyValueError::new_err(format!("could not decode image bytes: {err}"))
                })?
                .to_rgb8();
            let (width, height) = decoded.dimensions();
            let tensor = Tensor::from_vec(
                decoded.into_raw(),
                vec![i64::from(height), i64::from(width), 3],
                DType::Uint8,
            )
            .map_err(|err| PyValueError::new_err(format!("invalid decoded image: {err}")))?;
            Ok(Value::Tensor(tensor))
        }
        "t" => Ok(Value::Text(tuple.get_item(1)?.extract()?)),
        "n" => Ok(Value::Number(tuple.get_item(1)?.extract()?)),
        "l" => {
            let items = tuple.get_item(1)?;
            let list = items.cast::<PyList>()?;
            let mut out = Vec::with_capacity(list.len());
            for item in list.iter() {
                out.push(decode_value(&item)?);
            }
            Ok(Value::List(out))
        }
        "m" => {
            let entries = tuple.get_item(1)?;
            let dict = entries.cast::<PyDict>()?;
            let mut out: BTreeMap<String, Value> = BTreeMap::new();
            for (key, item) in dict.iter() {
                out.insert(key.extract()?, decode_value(&item)?);
            }
            Ok(Value::Map(out))
        }
        other => Err(PyValueError::new_err(format!(
            "unknown bridge value tag {other:?}"
        ))),
    }
}

/// Decode only the top-level observation entries the adapter reads.
///
/// Decoding the whole observation would let an unused — possibly
/// unencodable — env key abort a step the model does not even depend on.
fn decode_referenced_obs(
    encoded: &Bound<'_, PyAny>,
    referenced: &BTreeSet<String>,
) -> PyResult<BTreeMap<String, Value>> {
    let tuple = encoded.cast::<PyTuple>()?;
    let tag: String = tuple.get_item(0)?.extract()?;
    if tag != "m" {
        return Err(PyValueError::new_err(
            "expected a mapping observation".to_owned(),
        ));
    }
    // Referenced keys may be dotted (nested Dict paths); decode the whole
    // top-level entry that contains each.
    let top_level: BTreeSet<&str> = referenced
        .iter()
        .map(|key| key.split('.').next().unwrap_or(key.as_str()))
        .collect();
    let entries = tuple.get_item(1)?;
    let dict = entries.cast::<PyDict>()?;
    let mut out: BTreeMap<String, Value> = BTreeMap::new();
    for (key, value) in dict.iter() {
        let key: String = key.extract()?;
        if top_level.contains(key.as_str()) {
            out.insert(key, decode_value(&value)?);
        }
    }
    Ok(out)
}

fn encode_value<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    match value {
        Value::Tensor(tensor) => {
            let shape = PyTuple::new(py, tensor.shape().iter())?;
            let bytes = tensor.to_contiguous_bytes();
            let data = PyBytes::new(py, &bytes);
            Ok(PyTuple::new(
                py,
                [
                    "a".into_pyobject(py)?.into_any(),
                    tensor.dtype().name().into_pyobject(py)?.into_any(),
                    shape.into_any(),
                    data.into_any(),
                ],
            )?
            .into_any())
        }
        Value::Text(text) => Ok(PyTuple::new(
            py,
            [
                "t".into_pyobject(py)?.into_any(),
                text.into_pyobject(py)?.into_any(),
            ],
        )?
        .into_any()),
        Value::Number(number) => Ok(PyTuple::new(
            py,
            [
                "n".into_pyobject(py)?.into_any(),
                number.into_pyobject(py)?.into_any(),
            ],
        )?
        .into_any()),
        Value::List(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(encode_value(py, item)?)?;
            }
            Ok(PyTuple::new(py, ["l".into_pyobject(py)?.into_any(), list.into_any()])?.into_any())
        }
        Value::Map(entries) => {
            let dict = PyDict::new(py);
            for (key, item) in entries {
                dict.set_item(key, encode_value(py, item)?)?;
            }
            Ok(PyTuple::new(py, ["m".into_pyobject(py)?.into_any(), dict.into_any()])?.into_any())
        }
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
            let top = key.split('.').next().unwrap_or(&key).to_owned();
            keys.insert(top);
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

    /// Apply the observation plans to a bridge-encoded observation map.
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

    /// Apply the action plan to a bridge-encoded model action.
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

/// Resolve env annotations + spaces and a model spec into a plan handle.
///
/// `observation_space`/`action_space` are gymnasium space objects; they are
/// parsed and projected into the adapters `SpaceView`. Entrypoint trust is
/// passed through to the resolver (the Python wrapper decides it).
#[cfg_attr(
    feature = "stub-gen",
    gen_stub_pyfunction(
        module = "rlmesh._rlmesh",
        python = r#"
def adapters_resolve(env_annotations_json: str, observation_space: object, action_space: object, model_spec_json: str, trust_entrypoints: bool) -> AdapterPlan: ...
"#
    )
)]
#[pyfunction]
pub fn adapters_resolve(
    env_annotations_json: &str,
    observation_space: &Bound<'_, PyAny>,
    action_space: &Bound<'_, PyAny>,
    model_spec_json: &str,
    trust_entrypoints: bool,
) -> PyResult<PyAdapterPlan> {
    let annotations: EnvAnnotations = serde_json::from_str(env_annotations_json)
        .map_err(|err| PyValueError::new_err(format!("invalid env annotations: {err}")))?;
    let model_spec: ModelSpec = serde_json::from_str(model_spec_json)
        .map_err(|err| PyValueError::new_err(format!("invalid model spec: {err}")))?;
    let obs_view = SpaceView::from(&crate::spaces::parse_space(observation_space)?);
    let action_view = SpaceView::from(&crate::spaces::parse_space(action_space)?);
    let adapter = resolve(
        &annotations,
        &obs_view,
        &action_view,
        &model_spec,
        trust_entrypoints,
    )
    .map_err(|err| PyValueError::new_err(err.message))?;
    Ok(PyAdapterPlan { adapter })
}
