use std::sync::Mutex;

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rlmesh_spaces::v1::spaces::{
    BoxSpaceBuilder, DictSpaceBuilder, DiscreteBuilder, MultiBinaryBuilder, MultiDiscreteBuilder,
    SpaceSpec, TextBuilder, TupleSpaceBuilder,
};
use rlmesh_spaces::v1::{DType, EnvContract};

use super::sample::sample_space_value;
use super::spec_details::{space_kind_name, space_spec_details_to_py, space_spec_to_pydict};
use crate::spaces::utils::dtype_name;
use crate::spaces::{
    ValueBackend, make_space, meta_map_to_pydict, parse_space, py_any_to_space_value_with_backend,
};

#[gen_stub_pyclass]
#[pyclass(
    module = "rlmesh._rlmesh",
    name = "SpaceSpec",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PySpaceSpec {
    pub(super) inner: SpaceSpec,
}

#[gen_stub_pymethods]
#[pymethods]
impl PySpaceSpec {
    #[getter]
    fn kind(&self) -> &'static str {
        space_kind_name(&self.inner)
    }

    #[getter]
    fn shape(&self) -> Vec<i64> {
        self.inner.shape.clone()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        dtype_name(self.inner.dtype)
    }

    #[gen_stub(override_return_type(type_repr = "object", imports = ()))]
    fn _details<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        space_spec_details_to_py(py, &self.inner)
    }

    #[gen_stub(override_return_type(type_repr = "dict[str, object]", imports = ()))]
    fn _to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        space_spec_to_pydict(py, &self.inner)
    }

    #[gen_stub(override_return_type(type_repr = "Space", imports = ()))]
    fn to_space<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        Py::new(py, PySpace::new(self.inner.clone())).map(|value| value.into_any())
    }

    #[gen_stub(override_return_type(type_repr = "object", imports = ()))]
    fn to_gym_space<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        Ok(make_space(py, &self.inner)?.into_any().unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "SpaceSpec(kind={:?}, shape={:?}, dtype={:?})",
            self.kind(),
            self.inner.shape,
            self.dtype()
        )
    }
}

#[gen_stub_pyclass]
#[pyclass(
    module = "rlmesh._rlmesh",
    name = "EnvContract",
    frozen,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyEnvContract {
    inner: EnvContract,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyEnvContract {
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    #[getter]
    #[gen_stub(override_return_type(type_repr = "EnvContract", imports = ()))]
    fn env<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        env_contract_to_py(py, &self.inner)
    }

    #[getter]
    #[gen_stub(override_return_type(type_repr = "EnvContract", imports = ()))]
    fn spec<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        env_contract_to_py(py, &self.inner)
    }

    #[getter]
    fn render_mode(&self) -> Option<&str> {
        (!self.inner.render_mode.is_empty()).then_some(self.inner.render_mode.as_str())
    }

    #[getter]
    fn num_envs(&self) -> u32 {
        self.inner.num_envs
    }

    #[getter]
    #[gen_stub(override_return_type(type_repr = "dict[str, object] | None", imports = ()))]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        match self.inner.metadata.as_ref() {
            Some(metadata) => Ok(meta_map_to_pydict(py, metadata)?.into_any().unbind()),
            None => Ok(py.None()),
        }
    }

    #[getter]
    #[gen_stub(override_return_type(type_repr = "SpaceSpec", imports = ()))]
    fn observation_space<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        required_space_spec_to_py(
            py,
            self.inner.observation_space.as_ref(),
            "observation_space",
        )
    }

    #[getter]
    #[gen_stub(override_return_type(type_repr = "SpaceSpec", imports = ()))]
    fn action_space<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        required_space_spec_to_py(py, self.inner.action_space.as_ref(), "action_space")
    }

    #[gen_stub(override_return_type(type_repr = "dict[str, object]", imports = ()))]
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        result.set_item("id", self.inner.id.as_str())?;
        result.set_item("render_mode", self.render_mode())?;
        result.set_item("num_envs", self.inner.num_envs)?;
        result.set_item(
            "metadata",
            match self.inner.metadata.as_ref() {
                Some(metadata) => meta_map_to_pydict(py, metadata)?.into_any().unbind(),
                None => py.None(),
            },
        )?;
        let observation_space = self.inner.observation_space.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "remote environment contract missing observation_space",
            )
        })?;
        result.set_item(
            "observation_space",
            space_spec_to_pydict(py, observation_space)?
                .into_any()
                .unbind(),
        )?;
        let action_space = self.inner.action_space.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "remote environment contract missing action_space",
            )
        })?;
        result.set_item(
            "action_space",
            space_spec_to_pydict(py, action_space)?.into_any().unbind(),
        )?;
        Ok(result)
    }

    fn __repr__(&self) -> String {
        format!(
            "EnvContract(id={:?}, num_envs={})",
            self.inner.id, self.inner.num_envs
        )
    }
}

#[gen_stub_pyclass]
#[pyclass(module = "rlmesh._rlmesh", name = "Space")]
pub struct PySpace {
    spec: SpaceSpec,
    rng: Mutex<StdRng>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PySpace {
    #[getter]
    #[gen_stub(override_return_type(type_repr = "SpaceSpec", imports = ()))]
    fn spec<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        Py::new(
            py,
            PySpaceSpec {
                inner: self.spec.clone(),
            },
        )
        .map(|value| value.into_any())
    }

    #[getter]
    fn kind(&self) -> &'static str {
        space_kind_name(&self.spec)
    }

    #[getter]
    fn shape(&self) -> Vec<i64> {
        self.spec.shape.clone()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        dtype_name(self.spec.dtype)
    }

    fn seed(&self, seed: Option<u64>) -> Option<u64> {
        let seed = seed.unwrap_or_else(rand::random);
        *self.rng.lock().expect("rng mutex poisoned") = StdRng::seed_from_u64(seed);
        Some(seed)
    }

    #[gen_stub(override_return_type(type_repr = "object", imports = ()))]
    fn sample<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut rng = self.rng.lock().expect("rng mutex poisoned");
        sample_space_value(py, &self.spec, &mut rng)
    }

    fn contains(
        &self,
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "object", imports = ()))] value: &Bound<'_, PyAny>,
    ) -> bool {
        py_any_to_space_value_with_backend(py, value, &self.spec, ValueBackend::Native).is_ok()
    }

    fn __repr__(&self) -> String {
        format!(
            "Space(kind={:?}, shape={:?}, dtype={:?})",
            self.kind(),
            self.spec.shape,
            self.dtype()
        )
    }
}

impl PySpace {
    fn new(spec: SpaceSpec) -> Self {
        Self {
            spec,
            rng: Mutex::new(StdRng::seed_from_u64(rand::random())),
        }
    }
}

#[gen_stub_pyfunction(
    module = "rlmesh._rlmesh",
    python = r#"
def space_spec_from_gym_space(space: object) -> SpaceSpec: ...
"#
)]
#[pyfunction]
fn space_spec_from_gym_space(py: Python<'_>, space: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    Py::new(
        py,
        PySpaceSpec {
            inner: parse_space(space)?,
        },
    )
    .map(|value| value.into_any())
}

#[gen_stub_pyfunction(
    module = "rlmesh._rlmesh",
    python = r#"
def box_space_spec(low: float, high: float, shape: list[int], dtype: str | None = None) -> SpaceSpec: ...
"#
)]
#[pyfunction]
#[pyo3(signature = (low, high, shape, dtype=None))]
fn box_space_spec(
    py: Python<'_>,
    low: f64,
    high: f64,
    shape: Vec<i64>,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    space_spec_to_pyobject(
        py,
        BoxSpaceBuilder::scalar(low, high, shape)
            .dtype(parse_dtype(dtype, DType::Float32)?)
            .build(),
    )
}

#[gen_stub_pyfunction(
    module = "rlmesh._rlmesh",
    python = r#"
def discrete_space_spec(n: int, start: int = 0, dtype: str | None = None) -> SpaceSpec: ...
"#
)]
#[pyfunction]
#[pyo3(signature = (n, start=0, dtype=None))]
fn discrete_space_spec(
    py: Python<'_>,
    n: i64,
    start: i64,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    space_spec_to_pyobject(
        py,
        DiscreteBuilder::new(n)
            .start(start)
            .dtype(parse_dtype(dtype, DType::Int64)?)
            .build(),
    )
}

#[gen_stub_pyfunction(
    module = "rlmesh._rlmesh",
    python = r#"
def multi_binary_space_spec(shape: list[int], dtype: str | None = None) -> SpaceSpec: ...
"#
)]
#[pyfunction]
#[pyo3(signature = (shape, dtype=None))]
fn multi_binary_space_spec(
    py: Python<'_>,
    shape: Vec<i64>,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    space_spec_to_pyobject(
        py,
        MultiBinaryBuilder::shape(shape)
            .dtype(parse_dtype(dtype, DType::Uint8)?)
            .build(),
    )
}

#[gen_stub_pyfunction(
    module = "rlmesh._rlmesh",
    python = r#"
def multi_discrete_space_spec(nvec: list[int], dtype: str | None = None) -> SpaceSpec: ...
"#
)]
#[pyfunction]
#[pyo3(signature = (nvec, dtype=None))]
fn multi_discrete_space_spec(
    py: Python<'_>,
    nvec: Vec<i64>,
    dtype: Option<&str>,
) -> PyResult<Py<PyAny>> {
    space_spec_to_pyobject(
        py,
        MultiDiscreteBuilder::vector(nvec)
            .dtype(parse_dtype(dtype, DType::Int64)?)
            .build(),
    )
}

#[gen_stub_pyfunction(
    module = "rlmesh._rlmesh",
    python = r#"
def text_space_spec(max_length: int, min_length: int = 1, charset: str | None = None) -> SpaceSpec: ...
"#
)]
#[pyfunction]
#[pyo3(signature = (max_length, min_length=1, charset=None))]
fn text_space_spec(
    py: Python<'_>,
    max_length: i64,
    min_length: i64,
    charset: Option<String>,
) -> PyResult<Py<PyAny>> {
    let mut builder = TextBuilder::new(max_length).min_length(min_length);
    if let Some(charset) = charset {
        builder = builder.charset(charset);
    }
    space_spec_to_pyobject(py, builder.build())
}

#[gen_stub_pyfunction(
    module = "rlmesh._rlmesh",
    python = r#"
def dict_space_spec(entries: dict[str, object]) -> SpaceSpec: ...
"#
)]
#[pyfunction]
fn dict_space_spec(py: Python<'_>, entries: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let dict = entries.cast::<PyDict>()?;
    let mut builder = DictSpaceBuilder::new();
    for (key, value) in dict.iter() {
        let key = key.extract::<String>()?;
        let space = extract_space_spec(&value).ok_or_else(|| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "dict entry {key:?} is not an RLMesh space or SpaceSpec"
            ))
        })?;
        builder = builder.insert(key, space);
    }
    space_spec_to_pyobject(py, builder.build())
}

#[gen_stub_pyfunction(
    module = "rlmesh._rlmesh",
    python = r#"
def tuple_space_spec(spaces: list[object]) -> SpaceSpec: ...
"#
)]
#[pyfunction]
fn tuple_space_spec(py: Python<'_>, spaces: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let mut builder = TupleSpaceBuilder::new();
    for value in spaces.try_iter()? {
        let value = value?;
        let space = extract_space_spec(&value).ok_or_else(|| {
            pyo3::exceptions::PyTypeError::new_err("tuple item is not an RLMesh space or SpaceSpec")
        })?;
        builder = builder.with(space);
    }
    space_spec_to_pyobject(py, builder.build())
}

pub fn register_classes(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySpaceSpec>()?;
    module.add_class::<PyEnvContract>()?;
    module.add_class::<PySpace>()?;
    module.add_class::<super::tensor::PyTensor>()?;
    module.add_function(wrap_pyfunction!(space_spec_from_gym_space, module)?)?;
    module.add_function(wrap_pyfunction!(box_space_spec, module)?)?;
    module.add_function(wrap_pyfunction!(discrete_space_spec, module)?)?;
    module.add_function(wrap_pyfunction!(multi_binary_space_spec, module)?)?;
    module.add_function(wrap_pyfunction!(multi_discrete_space_spec, module)?)?;
    module.add_function(wrap_pyfunction!(text_space_spec, module)?)?;
    module.add_function(wrap_pyfunction!(dict_space_spec, module)?)?;
    module.add_function(wrap_pyfunction!(tuple_space_spec, module)?)?;
    Ok(())
}

pub fn env_contract_to_py<'py>(py: Python<'py>, env_contract: &EnvContract) -> PyResult<Py<PyAny>> {
    Py::new(
        py,
        PyEnvContract {
            inner: env_contract.clone(),
        },
    )
    .map(|value| value.into_any())
}

pub(crate) fn extract_space_spec(space: &Bound<'_, PyAny>) -> Option<SpaceSpec> {
    if let Ok(spec) = space.extract::<PyRef<'_, PySpaceSpec>>() {
        return Some(spec.inner.clone());
    }

    let spec_attr = space.getattr("spec").ok()?;
    spec_attr
        .extract::<PyRef<'_, PySpaceSpec>>()
        .ok()
        .map(|spec| spec.inner.clone())
}

fn space_spec_to_pyobject(
    py: Python<'_>,
    spec: Result<SpaceSpec, rlmesh_spaces::errors::SpaceError>,
) -> PyResult<Py<PyAny>> {
    let inner = spec.map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    Py::new(py, PySpaceSpec { inner }).map(|value| value.into_any())
}

fn parse_dtype(value: Option<&str>, default: DType) -> PyResult<DType> {
    let Some(value) = value else {
        return Ok(default);
    };
    match value.to_ascii_lowercase().as_str() {
        "bool" => Ok(DType::Bool),
        "uint8" => Ok(DType::Uint8),
        "int32" => Ok(DType::Int32),
        "int64" => Ok(DType::Int64),
        "float16" => Ok(DType::Float16),
        "float32" => Ok(DType::Float32),
        "float64" => Ok(DType::Float64),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported dtype {other:?}"
        ))),
    }
}

fn required_space_spec_to_py<'py>(
    py: Python<'py>,
    spec: Option<&SpaceSpec>,
    field: &'static str,
) -> PyResult<Py<PyAny>> {
    let spec = spec.ok_or_else(|| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "remote environment contract missing {field}"
        ))
    })?;
    Py::new(
        py,
        PySpaceSpec {
            inner: spec.clone(),
        },
    )
    .map(|value| value.into_any())
}

#[cfg(test)]
mod tests {
    use super::{PySpace, env_contract_to_py, register_classes};
    use pyo3::IntoPyObject;
    use pyo3::Python;
    use pyo3::types::{PyAnyMethods, PyDictMethods};
    use rlmesh_spaces::v1::EnvContract;
    use rlmesh_spaces::v1::spaces::{DiscreteBuilder, TextBuilder};

    #[test]
    fn converts_env_contract_to_native_python_object() {
        Python::attach(|py| {
            let module = pyo3::types::PyModule::new(py, "_rlmesh_test").unwrap();
            register_classes(&module).unwrap();

            let observation_space = TextBuilder::new(16).build().unwrap();
            let action_space = DiscreteBuilder::new(3).build().unwrap();
            let env_contract = EnvContract {
                id: "SpecViewEnv-v1".to_string(),
                observation_space: Some(observation_space),
                action_space: Some(action_space),
                num_envs: 1,
                ..Default::default()
            };

            let value = env_contract_to_py(py, &env_contract).unwrap();
            let mapping = value
                .bind(py)
                .call_method0("to_dict")
                .unwrap()
                .cast_into::<pyo3::types::PyDict>()
                .unwrap();
            assert_eq!(
                mapping
                    .get_item("id")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "SpecViewEnv-v1"
            );
            assert!(mapping.get_item("observation_space").unwrap().is_some());
            assert!(mapping.get_item("action_space").unwrap().is_some());
        });
    }

    #[test]
    fn env_contract_view_rejects_missing_required_spaces() {
        Python::attach(|py| {
            let module = pyo3::types::PyModule::new(py, "_rlmesh_test").unwrap();
            register_classes(&module).unwrap();

            let env_contract = EnvContract {
                id: "InvalidSpecViewEnv-v1".to_string(),
                num_envs: 1,
                ..Default::default()
            };

            let value = env_contract_to_py(py, &env_contract).unwrap();
            let err = value.bind(py).call_method0("to_dict").unwrap_err();
            assert!(err.to_string().contains("missing observation_space"));
        });
    }

    #[test]
    fn space_samples_rust_backed_values() {
        Python::attach(|py| {
            let spec = DiscreteBuilder::new(3).build().unwrap();
            let space = PySpace::new(spec);
            let sample = space.sample(py).unwrap();
            assert!(sample.extract::<i64>().is_ok());
        });
    }

    #[test]
    fn unrestricted_text_space_samples_contained_values() {
        Python::attach(|py| {
            let spec = TextBuilder::new(16).build().unwrap();
            let space = PySpace::new(spec);
            let sample = space.sample(py).unwrap();

            assert!(space.contains(py, &sample));
        });
    }

    #[test]
    fn space_contains_native_python_values() {
        Python::attach(|py| {
            let space = PySpace::new(DiscreteBuilder::new(3).build().unwrap());
            let valid = 2_i64.into_pyobject(py).unwrap();
            let invalid = 9_i64.into_pyobject(py).unwrap();

            assert!(space.contains(py, &valid.into_any()));
            assert!(!space.contains(py, &invalid.into_any()));
        });
    }
}
