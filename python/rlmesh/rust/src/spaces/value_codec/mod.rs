mod batch;
mod codec;
mod metadata;

use pyo3::prelude::*;

pub(crate) use self::batch::{
    batched_space_values_to_py_neutral, batched_space_values_to_py_with_backend,
    py_any_to_batched_space_values_with_backend,
};
pub(crate) use self::codec::{
    py_any_to_space_value_with_backend, space_value_to_py_neutral, space_value_to_py_with_backend,
    tensor_from_shape,
};
pub(crate) use self::metadata::{meta_map_to_pydict, py_any_to_meta_map};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ValueBackend {
    Auto,
    Native,
}

impl ValueBackend {
    pub fn prefers_numpy(self, py: Python<'_>) -> PyResult<bool> {
        match self {
            Self::Native => Ok(false),
            Self::Auto => Ok(py.import("numpy").is_ok()),
        }
    }
}
