mod array_codec;
mod batch;
mod metadata;
mod native_value_codec;
mod neutral_value_codec;

use pyo3::prelude::*;

pub(crate) use self::batch::{
    batched_space_values_to_py_with_backend, py_any_to_batched_space_values_with_backend,
};
pub(crate) use self::metadata::{meta_map_to_pydict, py_any_to_meta_map};
pub(crate) use self::native_value_codec::{
    py_any_to_space_value_with_backend, space_value_to_py_with_backend,
};
pub(crate) use self::neutral_value_codec::{
    batched_space_values_to_py_neutral, space_value_to_py_neutral, tensor_from_shape,
};

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
