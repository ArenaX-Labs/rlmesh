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
        let mut stacklevel = 1;
        if let Ok(mut frame) = py
            .import("sys")
            .and_then(|sys| sys.call_method1("_getframe", (0,)))
        {
            loop {
                let module = frame
                    .getattr("f_globals")
                    .and_then(|globals| globals.get_item("__name__"))
                    .and_then(|name| name.extract::<String>())
                    .unwrap_or_default();
                if module != "rlmesh" && !module.starts_with("rlmesh.") {
                    break;
                }
                stacklevel += 1;
                frame = frame.getattr("f_back")?;
                if frame.is_none() {
                    break;
                }
            }
        }
        PyErr::warn(py, &category(py, warning.category), &message, stacklevel)
    })
    .map_err(|err| {
        rlmesh::binding::AttachedError::attach(rlmesh::Error::Internal(err.to_string()), err)
    })
}

/// Install the bridge as the process's native warning handler.
pub(crate) fn install() {
    rlmesh::set_warning_handler(Some(Box::new(warn)));
}
