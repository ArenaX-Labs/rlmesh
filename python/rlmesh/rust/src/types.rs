use pyo3::create_exception;
use pyo3::exceptions::{
    PyConnectionError, PyKeyboardInterrupt, PyOSError, PyRuntimeError, PyTimeoutError, PyTypeError,
    PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rlmesh::{
    EnvironmentError, Error, ErrorCode, InvalidArgumentError, ModelError, ModelErrorCode,
};
use rlmesh_spaces::SpaceValue;

create_exception!(
    _rlmesh,
    RLMeshException,
    PyRuntimeError,
    "Base exception for RLMesh errors"
);

create_exception!(
    _rlmesh,
    ProtocolException,
    RLMeshException,
    "Protocol-level errors (generation mismatch, invalid messages)"
);

create_exception!(
    _rlmesh,
    EnvironmentException,
    RLMeshException,
    "Environment-level errors (from the spaces environment)"
);

create_exception!(
    _rlmesh,
    RecoverableEnvironmentException,
    EnvironmentException,
    "An environment error that fails one request but leaves the session usable"
);

create_exception!(
    _rlmesh,
    ModelException,
    RLMeshException,
    "A failure the model reported; carries `code` and `is_recoverable`.\n\n\
     Raise `ModelException(message, is_recoverable=True)` from a model to fail \
     one request as recoverable."
);

const MODEL_EXCEPTION_INIT: &std::ffi::CStr = cr#"
def __init__(self, message, *, is_recoverable=False):
    RLMeshException.__init__(self, message)
    self.is_recoverable = bool(is_recoverable)

__init__.__qualname__ = "ModelException.__init__"
ModelException.__init__ = __init__
"#;

const CODED_EXCEPTION_STR: &std::ffi::CStr = cr#"
def __str__(self):
    return f"[{self.code}] {BaseException.__str__(self)}"

__str__.__qualname__ = "EnvironmentException.__str__"
EnvironmentException.__str__ = __str__
ModelException.__str__ = __str__
"#;

/// The message an environment or model exception was raised with, without
/// its `[CODE]` prefix (`BaseException.__str__`); `str(exc)` for any other.
pub(crate) fn raw_exception_text(value: &Bound<'_, PyAny>) -> String {
    let py = value.py();
    let raw = if value.is_instance_of::<EnvironmentException>()
        || value.is_instance_of::<ModelException>()
    {
        py.get_type::<pyo3::exceptions::PyBaseException>()
            .call_method1("__str__", (value,))
            .and_then(|text| text.extract::<String>())
    } else {
        value.str().map(|text| text.to_string())
    };
    raw.unwrap_or_default()
}

/// The wire message of an author's exception: an environment or
/// model exception's raw message, `str(exc)` for any other; no phase or type
/// name, since the receiver adds `[CODE] ` once.
pub(crate) fn py_err_text(err: &PyErr) -> String {
    Python::attach(|py| raw_exception_text(err.value(py)))
}

/// Convert rlmesh facade Error to Python exception.
pub fn to_py_err(err: Error) -> PyErr {
    match err {
        Error::Attached(attached) => Python::attach(|py| {
            attached
                .payload::<PyErr>()
                .map(|error| error.clone_ref(py))
                .unwrap_or_else(|| to_py_err(attached.error().clone()))
        }),
        Error::Internal(message) => RLMeshException::new_err(message),
        Error::Protocol(message) => ProtocolException::new_err(message),
        Error::Environment(e) => env_error_to_py(e),
        Error::Model(e) => model_error_to_py(e),
        Error::InvalidArgument(e) => invalid_argument_to_py(e),
        Error::Type(message) => PyTypeError::new_err(message),
        Error::Runtime(message) => PyRuntimeError::new_err(message),
        Error::Os(message) => PyOSError::new_err(message),
        Error::Connection(message) => PyConnectionError::new_err(message),
        error @ Error::Timeout(_) => PyTimeoutError::new_err(error.to_string()),
        error @ Error::Cancelled => PyKeyboardInterrupt::new_err(error.to_string()),
        other => RLMeshException::new_err(other.to_string()),
    }
}

const TYPED_EXCEPTION: &std::ffi::CStr = cr#"
import builtins

_typed = {}

def typed(base, name):
    key = (base, name)
    if key not in _typed:
        builtin = getattr(builtins, name, None)
        cls = None
        if isinstance(builtin, type) and issubclass(builtin, Exception):
            try:
                cls = type(
                    name,
                    (base, builtin),
                    {"__str__": BaseException.__str__, "__module__": base.__module__},
                )
            except TypeError:
                pass
        _typed[key] = cls
    return _typed[key]
"#;

static TYPED: pyo3::sync::PyOnceLock<Py<PyAny>> = pyo3::sync::PyOnceLock::new();

fn typed_exception<'py>(
    py: Python<'py>,
    base: Bound<'py, pyo3::types::PyType>,
    class: &str,
    message: &str,
) -> Option<Bound<'py, PyAny>> {
    let typed = TYPED
        .get_or_try_init(py, || -> PyResult<Py<PyAny>> {
            let globals = PyDict::new(py);
            py.run(TYPED_EXCEPTION, Some(&globals), None)?;
            Ok(globals.get_item("typed")?.expect("defined").unbind())
        })
        .ok()?;
    let cls = typed.bind(py).call1((base, class)).ok()?;
    if cls.is_none() {
        return None;
    }
    let (phase, message) = match message.split_once(" failed: ") {
        Some((phase, rest)) if !phase.contains(' ') => (Some(phase), rest),
        _ => (None, message),
    };
    let exception = cls.call1((message,)).ok()?;
    if let Some(phase) = phase {
        exception.setattr("phase", phase).ok()?;
        exception
            .call_method1("add_note", (format!("raised by the env's {phase}"),))
            .ok()?;
    }
    Some(exception)
}

fn env_error_to_py(err: EnvironmentError) -> PyErr {
    let msg = err.message.clone();
    let typed = err.class.as_deref().and_then(|class| {
        Python::attach(|py| {
            let base = if err.is_recoverable {
                py.get_type::<RecoverableEnvironmentException>()
            } else {
                py.get_type::<EnvironmentException>()
            };
            typed_exception(py, base, class, &msg).map(PyErr::from_value)
        })
    });
    let py_err = typed.unwrap_or_else(|| {
        if err.is_recoverable {
            RecoverableEnvironmentException::new_err(msg)
        } else {
            EnvironmentException::new_err(msg)
        }
    });
    annotate(&py_err, err.code.as_str(), err.is_recoverable);
    py_err
}

fn model_error_to_py(err: ModelError) -> PyErr {
    let py_err = ModelException::new_err(err.message.clone());
    annotate(&py_err, err.code.as_str(), err.is_recoverable);
    py_err
}

fn annotate(py_err: &PyErr, code: &str, is_recoverable: bool) {
    Python::attach(|py| {
        let value = py_err.value(py);
        // A failure only loses the annotation, never the error.
        let _ = value.setattr("code", code);
        let _ = value.setattr("is_recoverable", is_recoverable);
    });
}

fn invalid_argument_to_py(err: InvalidArgumentError) -> PyErr {
    if err.adapter_resolution.is_some() {
        let adapter_error = Python::attach(|py| {
            py.import("rlmesh.adapters")
                .and_then(|module| module.getattr("AdapterResolutionError"))
                .and_then(|class| class.call1((err.message.as_str(),)))
                .map(PyErr::from_value)
                .ok()
        });
        if let Some(py_err) = adapter_error {
            return py_err;
        }
    }
    PyValueError::new_err(err.message)
}

/// The facade error a Python exception raised while resolving a route becomes: an
/// `AdapterResolutionError` stays one; anything else is an internal failure.
pub(crate) fn resolve_callback_error(err: &PyErr) -> Error {
    Python::attach(|py| {
        let value = err.value(py);
        let adapter = py
            .import("rlmesh.adapters")
            .and_then(|module| module.getattr("AdapterResolutionError"))
            .is_ok_and(|class| value.is_instance(&class).unwrap_or(false));
        if adapter {
            Error::adapter_resolution(raw_exception_text(value))
        } else {
            Error::Internal(err.to_string())
        }
    })
}

const SERVED_TRACEBACK_FRAMES: usize = 8;
const SERVED_TRACEBACK_BYTES: usize = 4096;

fn model_frames(py: Python<'_>, tb: &Bound<'_, pyo3::types::PyTraceback>) -> PyResult<Vec<String>> {
    let traceback = py.import("traceback")?;
    let package = py
        .import("rlmesh")?
        .getattr("__file__")?
        .extract::<String>()?;
    let package = std::path::Path::new(&package)
        .parent()
        .map(|dir| format!("{}{}", dir.display(), std::path::MAIN_SEPARATOR))
        .unwrap_or_default();
    let summary = traceback.call_method1("extract_tb", (tb,))?;
    let frames: Vec<Bound<'_, PyAny>> = summary.try_iter()?.collect::<PyResult<_>>()?;
    let first = frames
        .iter()
        .position(|frame| {
            frame
                .getattr("filename")
                .and_then(|name| name.extract::<String>())
                .map_or(true, |name| {
                    package.is_empty() || !name.starts_with(&package)
                })
        })
        .unwrap_or(0);
    let kept = pyo3::types::PyList::new(py, &frames[first..])?;
    traceback.call_method1("format_list", (kept,))?.extract()
}

fn foreign_exception_text(err: &PyErr) -> String {
    Python::attach(|py| {
        let value = err.value(py);
        let headline = py
            .import("traceback")
            .and_then(|traceback| traceback.call_method1("format_exception_only", (value,)))
            .and_then(|lines| lines.extract::<Vec<String>>())
            .map(|lines| lines.concat().trim_end().to_string())
            .unwrap_or_else(|_| err.to_string());
        let frames = err
            .traceback(py)
            .and_then(|tb| model_frames(py, &tb).ok())
            .unwrap_or_default();
        let mut text = headline;
        if !frames.is_empty() {
            let kept = &frames[frames.len().saturating_sub(SERVED_TRACEBACK_FRAMES)..];
            let mut tail = kept.concat();
            if tail.len() > SERVED_TRACEBACK_BYTES {
                let mut start = tail.len() - SERVED_TRACEBACK_BYTES;
                while !tail.is_char_boundary(start) {
                    start += 1;
                }
                tail = format!("  …\n{}", &tail[start..]);
            } else if kept.len() < frames.len() {
                tail = format!("  …\n{tail}");
            }
            text.push_str("\nTraceback (most recent call last):\n");
            text.push_str(tail.trim_end());
        }
        rlmesh_session::address::redact_text(&text)
    })
}

/// The facade error a Python exception raised by a model callback becomes: a `ModelException`
/// keeps its recoverable flag as a model failure with code `INTERNAL`; anything else is an
/// internal failure with the exception type, message and traceback tail ([`foreign_exception_text`]).
pub(crate) fn model_callback_error(err: &PyErr) -> Error {
    Python::attach(|py| {
        let value = err.value(py);
        if value.is_instance_of::<ModelException>() {
            let message = raw_exception_text(value);
            let is_recoverable = value
                .getattr("is_recoverable")
                .and_then(|flag| flag.is_truthy())
                .unwrap_or(false);
            Error::Model(ModelError {
                code: ModelErrorCode::Internal,
                message,
                is_recoverable,
            })
        } else {
            Error::Internal(foreign_exception_text(err))
        }
    })
}

/// Register exception types with the Python module.
pub fn register_exceptions(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("RLMeshException", py.get_type::<RLMeshException>())?;
    m.add("ProtocolException", py.get_type::<ProtocolException>())?;
    let environment = py.get_type::<EnvironmentException>();
    environment.setattr("code", ErrorCode::Internal.as_str())?;
    environment.setattr("is_recoverable", false)?;
    m.add("EnvironmentException", &environment)?;
    let recoverable = py.get_type::<RecoverableEnvironmentException>();
    recoverable.setattr("is_recoverable", true)?;
    m.add("RecoverableEnvironmentException", recoverable)?;
    let model = py.get_type::<ModelException>();
    model.setattr("code", ModelErrorCode::Internal.as_str())?;
    model.setattr("is_recoverable", false)?;
    let globals = PyDict::new(py);
    globals.set_item("RLMeshException", py.get_type::<RLMeshException>())?;
    globals.set_item("ModelException", &model)?;
    py.run(MODEL_EXCEPTION_INIT, Some(&globals), None)?;
    globals.set_item("EnvironmentException", &environment)?;
    py.run(CODED_EXCEPTION_STR, Some(&globals), None)?;
    m.add("ModelException", model)?;
    Ok(())
}

/// Byte size of a single space value's payload (profiling).
pub(crate) fn space_value_size(value: &SpaceValue) -> usize {
    match value {
        SpaceValue::Box(value) => value.nbytes(),
        SpaceValue::Discrete(_) => std::mem::size_of::<i64>(),
        SpaceValue::MultiBinary(values) => values.len(),
        SpaceValue::MultiDiscrete(values) => values.len() * std::mem::size_of::<i64>(),
        SpaceValue::Text(value) => value.len(),
        SpaceValue::Dict(values) => values.values().map(space_value_size).sum(),
        SpaceValue::Tuple(values) => values.iter().map(space_value_size).sum(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_error(code: ErrorCode, is_recoverable: bool) -> Error {
        Error::Environment(EnvironmentError {
            code,
            message: "boom".into(),
            is_recoverable,
            class: None,
        })
    }

    fn raised(err: Error) -> (String, String, Option<String>, Option<bool>) {
        let py_err = to_py_err(err);
        Python::attach(|py| {
            let module = PyModule::new(py, "_rlmesh_test").unwrap();
            register_exceptions(&module).unwrap();
            let value = py_err.value(py);
            let class = value.get_type().name().unwrap().to_string();
            let text = value.str().unwrap().to_string();
            let code = value
                .getattr("code")
                .ok()
                .map(|code| code.extract::<String>().unwrap());
            let recoverable = value
                .getattr("is_recoverable")
                .ok()
                .map(|flag| flag.extract::<bool>().unwrap());
            (class, text, code, recoverable)
        })
    }

    #[test]
    fn every_env_code_raises_the_environment_family() {
        for code in [
            ErrorCode::Timeout,
            ErrorCode::InvalidAction,
            ErrorCode::Cancelled,
            ErrorCode::Unsupported,
            ErrorCode::Busy,
        ] {
            let (class, text, got, recoverable) = raised(env_error(code, false));
            assert_eq!(class, "EnvironmentException", "{code}");
            assert_eq!(text, format!("[{code}] boom"));
            assert_eq!(got.as_deref(), Some(code.as_str()));
            assert_eq!(recoverable, Some(false));
            let (class, ..) = raised(env_error(code, true));
            assert_eq!(class, "RecoverableEnvironmentException", "{code}");
        }
    }

    #[test]
    fn model_failures_raise_model_exception() {
        let (class, text, code, recoverable) = raised(Error::model_recoverable("declined"));
        assert_eq!(class, "ModelException");
        assert_eq!(text, "[INTERNAL] declined");
        assert_eq!(code.as_deref(), Some("INTERNAL"));
        assert_eq!(recoverable, Some(true));
        Python::attach(|py| {
            let value = to_py_err(Error::model("x")).into_value(py);
            let args = value.bind(py).getattr("args").unwrap();
            assert_eq!(args.get_item(0).unwrap().extract::<String>().unwrap(), "x");
            assert!(value.bind(py).is_instance_of::<RLMeshException>());
            assert!(value.bind(py).is_instance_of::<PyRuntimeError>());
        });
    }

    #[test]
    fn each_variant_raises_its_python_class() {
        let cases = [
            (Error::Internal("boom".into()), "RLMeshException", "boom"),
            (Error::Protocol("bad".into()), "ProtocolException", "bad"),
            (Error::invalid_argument("bad"), "ValueError", "bad"),
            (Error::Type("kind".into()), "TypeError", "kind"),
            (Error::Runtime("closed".into()), "RuntimeError", "closed"),
            (Error::Os("exists".into()), "OSError", "exists"),
            (
                Error::Connection("reset".into()),
                "ConnectionError",
                "reset",
            ),
            (
                Error::Timeout(std::time::Duration::from_secs(5)),
                "TimeoutError",
                "operation timed out after 5s",
            ),
            (Error::Cancelled, "KeyboardInterrupt", "cancelled"),
        ];
        for (error, class, text) in cases {
            let (got_class, got_text, ..) = raised(error);
            assert_eq!((got_class.as_str(), got_text.as_str()), (class, text));
        }
    }

    #[test]
    fn env_error_code_formats_with_label() {
        assert_eq!(ErrorCode::Timeout.to_string(), "TIMEOUT");
        assert_eq!(ErrorCode::InvalidAction.to_string(), "INVALID_ACTION");
        assert_eq!(ErrorCode::Closed.to_string(), "CLOSED");
    }
}
