//! Bindings for the `rlmesh-adapters` core: spec resolution and plan
//! application.
//!
//! Values cross the boundary in a small tagged-tuple encoding produced by
//! `rlmesh.adapters.helpers.bridge`:
//!
//! - `("a", dtype, shape, bytes)` — a dense array (native byte order)
//! - `("b", bytes)` — an encoded image (PNG/JPEG), decoded here to an
//!   RGB uint8 HWC array via the `image` crate (codec-level bridge
//!   behavior, deliberately not part of the pinned v1 semantics)
//! - `("t", str)` — text
//! - `("n", float)` — a scalar number
//! - `("l", [encoded, ...])` — a list
//! - `("m", {key: encoded})` — a nested mapping
//!
//! Custom inputs are never evaluated here: the plan keeps them as holes
//! ([`SkipCustoms`]) and the Python wrapper runs the user's callable on
//! the raw Python observation afterwards.

use std::collections::BTreeMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use rlmesh_adapters::v1::{
    Array, ArrayData, Dtype, EnvIoSpec, ModelIoSpec, ObsPlan, ResolvedAdapter, SkipCustoms, Value,
    resolve, roles,
};

/// Wire-vocabulary constants re-exported to Python. The `rlmesh-adapters`
/// crate is the single source of truth: bindings re-export, never
/// re-declare. (C++ bindings will consume the crate constants directly.)
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

/// Register the wire-vocabulary constants on the `_rlmesh` module.
pub fn register_constants(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for (name, value) in WIRE_CONSTANTS {
        m.add(name, *value)?;
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
            let dtype = Dtype::parse(&tuple.get_item(1)?.extract::<String>()?)
                .map_err(|err| PyValueError::new_err(err.message))?;
            let shape: Vec<usize> = tuple.get_item(2)?.extract()?;
            let raw: Vec<u8> = tuple.get_item(3)?.extract()?;
            let data = match dtype {
                Dtype::U8 => ArrayData::U8(raw),
                Dtype::I32 => ArrayData::I32(
                    raw.chunks_exact(4)
                        .map(|chunk| i32::from_ne_bytes(chunk.try_into().unwrap()))
                        .collect(),
                ),
                Dtype::I64 => ArrayData::I64(
                    raw.chunks_exact(8)
                        .map(|chunk| i64::from_ne_bytes(chunk.try_into().unwrap()))
                        .collect(),
                ),
                Dtype::F32 => ArrayData::F32(
                    raw.chunks_exact(4)
                        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
                        .collect(),
                ),
                Dtype::F64 => ArrayData::F64(
                    raw.chunks_exact(8)
                        .map(|chunk| f64::from_ne_bytes(chunk.try_into().unwrap()))
                        .collect(),
                ),
            };
            Ok(Value::Array(Array { dtype, shape, data }))
        }
        "b" => {
            let raw: Vec<u8> = tuple.get_item(1)?.extract()?;
            let decoded = image::load_from_memory(&raw)
                .map_err(|err| {
                    PyValueError::new_err(format!("could not decode image bytes: {err}"))
                })?
                .to_rgb8();
            let (width, height) = decoded.dimensions();
            Ok(Value::Array(Array {
                dtype: Dtype::U8,
                shape: vec![height as usize, width as usize, 3],
                data: ArrayData::U8(decoded.into_raw()),
            }))
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

fn array_bytes(data: &ArrayData) -> Vec<u8> {
    match data {
        ArrayData::U8(values) => values.clone(),
        ArrayData::I32(values) => values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect(),
        ArrayData::I64(values) => values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect(),
        ArrayData::F32(values) => values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect(),
        ArrayData::F64(values) => values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect(),
    }
}

fn encode_value<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    match value {
        Value::Array(array) => {
            let shape = PyTuple::new(py, array.shape.iter())?;
            let data = PyBytes::new(py, &array_bytes(&array.data));
            Ok(PyTuple::new(
                py,
                [
                    "a".into_pyobject(py)?.into_any(),
                    array.dtype.as_str().into_pyobject(py)?.into_any(),
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
#[gen_stub_pyclass]
#[pyclass(module = "rlmesh._rlmesh", name = "AdapterPlan", frozen)]
pub struct PyAdapterPlan {
    adapter: ResolvedAdapter,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyAdapterPlan {
    /// Human-readable summary of the resolved transformations.
    fn describe(&self) -> String {
        self.adapter.describe()
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
        let Value::Map(raw_obs) = decode_value(raw_obs)? else {
            return Err(PyValueError::new_err(
                "expected a mapping observation".to_owned(),
            ));
        };
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
        encode_value(py, &Value::Array(action))
    }
}

/// Resolve serialized env/model specs into an adapter plan handle.
///
/// Entrypoint trust is enforced by the Python wrapper before this call,
/// so custom inputs are always admitted here as plan holes.
#[gen_stub_pyfunction(
    module = "rlmesh._rlmesh",
    python = r#"
def adapters_resolve(env_spec_json: str, model_spec_json: str) -> AdapterPlan: ...
"#
)]
#[pyfunction]
pub fn adapters_resolve(env_spec_json: &str, model_spec_json: &str) -> PyResult<PyAdapterPlan> {
    let env_spec: EnvIoSpec = serde_json::from_str(env_spec_json)
        .map_err(|err| PyValueError::new_err(format!("invalid env spec: {err}")))?;
    let model_spec: ModelIoSpec = serde_json::from_str(model_spec_json)
        .map_err(|err| PyValueError::new_err(format!("invalid model spec: {err}")))?;
    let adapter =
        resolve(&env_spec, &model_spec, true).map_err(|err| PyValueError::new_err(err.message))?;
    Ok(PyAdapterPlan { adapter })
}
