//! Route the native warnings (R-23) through Python's `warnings.warn`.
//!
//! The engine behind `Model.run` and `Model.serve` warns through the facade's
//! handler; in a Python process those warnings are Python warnings, filtered
//! and shown (or raised under `-W error`) by Python's own machinery.

use std::ffi::CString;

use pyo3::exceptions::{PyRuntimeWarning, PyUserWarning};
use pyo3::prelude::*;
use pyo3::types::PyType;
use rlmesh::{Warning, WarningCategory};

fn category<'py>(py: Python<'py>, category: WarningCategory) -> Bound<'py, PyType> {
    match category {
        WarningCategory::RuntimeWarning => py.get_type::<PyRuntimeWarning>(),
        WarningCategory::WorkflowEditionWarning => py
            .import("rlmesh._editions")
            .and_then(|module| module.getattr("WorkflowEditionWarning"))
            .and_then(|class| class.cast_into::<PyType>().map_err(PyErr::from))
            .unwrap_or_else(|_| py.get_type::<PyUserWarning>()),
        _ => py.get_type::<PyUserWarning>(),
    }
}

fn warn(warning: &Warning) -> rlmesh::Result<()> {
    Python::attach(|py| {
        let message = CString::new(warning.message.replace('\0', " "))
            .unwrap_or_else(|_| CString::from(c"rlmesh warning"));
        PyErr::warn(py, &category(py, warning.category), &message, 1)
    })
    .map_err(|err| rlmesh::Error::Internal(err.to_string()))
}

/// Install the bridge as the process's native warning handler.
pub(crate) fn install() {
    rlmesh::set_warning_handler(Some(Box::new(warn)));
}
