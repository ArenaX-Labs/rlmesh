use crate::spaces::composite::*;
use crate::spaces::fundamental::*;
use crate::spaces::spec_view::extract_space_spec;
use crate::spaces::utils::*;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use rlmesh_spaces::spaces::*;

pub fn parse_space<'py>(space: &Bound<'py, PyAny>) -> PyResult<SpaceSpec> {
    if let Some(spec) = extract_space_spec(space) {
        return Ok(spec);
    }

    // Dispatch with isinstance against the imported Gymnasium classes (rather
    // than the unqualified class name) so user subclasses such as
    // `class MyBox(gym.spaces.Box)` are recognised, consistent with the
    // Python-side from_gymnasium_space converter.
    let spaces = import_gym(space.py())?.getattr("spaces")?;

    if is_gym_instance(space, &spaces, "Box")? {
        parse_box(space)
    } else if is_gym_instance(space, &spaces, "Discrete")? {
        parse_discrete(space)
    } else if is_gym_instance(space, &spaces, "MultiBinary")? {
        parse_multibinary(space)
    } else if is_gym_instance(space, &spaces, "MultiDiscrete")? {
        parse_multidiscrete(space)
    } else if is_gym_instance(space, &spaces, "Text")? {
        parse_text(space)
    } else if is_gym_instance(space, &spaces, "Dict")? {
        parse_dict(space)
    } else if is_gym_instance(space, &spaces, "Tuple")? {
        parse_tuple(space)
    } else {
        let type_name = space.get_type().name()?.to_string();
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Unsupported space type: {type_name}"
        )))
    }
}

/// Whether `space` is an instance of `gymnasium.spaces.<name>`.
///
/// A missing class (e.g. older Gym lacking `Text`) is treated as "not an
/// instance" rather than an error.
fn is_gym_instance(
    space: &Bound<'_, PyAny>,
    spaces: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<bool> {
    match spaces.getattr(name) {
        Ok(class) => space.is_instance(&class),
        Err(_) => Ok(false),
    }
}

pub fn make_space<'py>(py: Python<'py>, space: &SpaceSpec) -> PyResult<Bound<'py, PyAny>> {
    let spaces = import_gym(py)?.getattr("spaces")?;
    match space
        .spec
        .as_ref()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("spec missing"))?
    {
        // === Fundamental === //
        SpaceKind::Box(_) => make_box(py, &spaces, space),
        SpaceKind::Discrete(_) => make_discrete(py, &spaces, space),
        SpaceKind::MultiBinary(_) => make_multibinary(py, &spaces, space),
        SpaceKind::MultiDiscrete(_) => make_multidiscrete(py, &spaces, space),
        SpaceKind::Text(_) => make_text(py, &spaces, space),

        // === Composite === //
        SpaceKind::Dict(_) => make_dict(py, &spaces, space),
        SpaceKind::Tuple(_) => make_tuple(py, &spaces, space),
    }
}

#[cfg(test)]
mod tests {
    use super::{make_space, parse_space};
    use crate::spaces::register_classes;
    use crate::spaces::utils::import_gym;
    use pyo3::Bound;
    use pyo3::Python;
    use pyo3::types::{PyAny, PyAnyMethods, PyDict, PyDictMethods, PyModule};
    use rlmesh_spaces::spaces::{SpaceKind, TextBuilder};

    fn discrete_space<'py>(py: Python<'py>, n: i64) -> Bound<'py, PyAny> {
        import_gym(py)
            .unwrap()
            .getattr("spaces")
            .unwrap()
            .getattr("Discrete")
            .unwrap()
            .call1((n,))
            .unwrap()
    }

    #[test]
    fn parse_space_accepts_native_space_spec_objects() {
        Python::attach(|py| {
            let module = PyModule::new(py, "_rlmesh_space_test").unwrap();
            register_classes(&module).unwrap();

            let spec_obj = module
                .getattr("space_spec_from_gym_space")
                .unwrap()
                .call1((discrete_space(py, 3),))
                .unwrap();

            let parsed = parse_space(&spec_obj).unwrap();
            assert!(matches!(parsed.spec, Some(SpaceKind::Discrete(_))));
        });
    }

    #[test]
    fn parse_space_accepts_objects_exposing_spec() {
        Python::attach(|py| {
            let module = PyModule::new(py, "_rlmesh_space_test").unwrap();
            register_classes(&module).unwrap();

            let spec = module
                .getattr("space_spec_from_gym_space")
                .unwrap()
                .call1((discrete_space(py, 4),))
                .unwrap();

            let kwargs = PyDict::new(py);
            kwargs.set_item("spec", spec).unwrap();
            let native_like = py
                .import("types")
                .unwrap()
                .getattr("SimpleNamespace")
                .unwrap()
                .call((), Some(&kwargs))
                .unwrap();

            let parsed = parse_space(&native_like).unwrap();
            assert!(matches!(parsed.spec, Some(SpaceKind::Discrete(_))));
        });
    }

    #[test]
    fn parse_space_accepts_native_space_wrappers_via_spec() {
        Python::attach(|py| {
            let module = PyModule::new(py, "_rlmesh_space_test").unwrap();
            register_classes(&module).unwrap();

            let spec = module
                .getattr("space_spec_from_gym_space")
                .unwrap()
                .call1((discrete_space(py, 5),))
                .unwrap();
            let native = spec.call_method0("to_space").unwrap();

            let parsed = parse_space(&native).unwrap();
            assert!(matches!(parsed.spec, Some(SpaceKind::Discrete(_))));
        });
    }

    #[test]
    fn parse_space_still_accepts_gym_spaces() {
        Python::attach(|py| {
            let discrete = discrete_space(py, 6);

            let parsed = parse_space(&discrete).unwrap();
            assert!(matches!(parsed.spec, Some(SpaceKind::Discrete(_))));
        });
    }

    #[test]
    fn parse_space_accepts_gym_space_subclasses() {
        Python::attach(|py| {
            let spaces = import_gym(py).unwrap().getattr("spaces").unwrap();
            let globals = PyDict::new(py);
            globals.set_item("spaces", &spaces).unwrap();
            // A user-defined subclass of gym.spaces.Box, as found in many
            // wrapper/robotics libraries.
            py.run(
                pyo3::ffi::c_str!("class MyBox(spaces.Box):\n    pass\n"),
                Some(&globals),
                None,
            )
            .unwrap();
            let my_box = py
                .eval(
                    pyo3::ffi::c_str!("MyBox(low=-1.0, high=1.0, shape=(2,))"),
                    Some(&globals),
                    None,
                )
                .unwrap();

            let parsed = parse_space(&my_box).unwrap();
            assert!(matches!(parsed.spec, Some(SpaceKind::Box(_))));
        });
    }

    #[test]
    fn parse_space_treats_default_gym_text_charset_as_unrestricted() {
        Python::attach(|py| {
            let spaces = import_gym(py).unwrap().getattr("spaces").unwrap();
            let text = spaces.getattr("Text").unwrap().call1((32,)).unwrap();

            assert!(
                !text
                    .call_method1("contains", ("pick up the object!",))
                    .unwrap()
                    .extract::<bool>()
                    .unwrap()
            );

            let parsed = parse_space(&text).unwrap();
            let Some(SpaceKind::Text(spec)) = parsed.spec else {
                panic!("expected Text space");
            };

            assert_eq!(spec.max_length, 32);
            assert_eq!(spec.charset, "");
        });
    }

    #[test]
    fn parse_space_preserves_gym_text_charset() {
        Python::attach(|py| {
            let spaces = import_gym(py).unwrap().getattr("spaces").unwrap();
            let kwargs = PyDict::new(py);
            kwargs.set_item("charset", "ab ").unwrap();
            let text = spaces
                .getattr("Text")
                .unwrap()
                .call((12,), Some(&kwargs))
                .unwrap();

            let parsed = parse_space(&text).unwrap();
            let Some(SpaceKind::Text(spec)) = parsed.spec else {
                panic!("expected Text space");
            };

            assert_eq!(spec.max_length, 12);
            assert_eq!(spec.charset, " ab");
        });
    }

    #[test]
    fn make_space_unrestricted_text_uses_printable_gym_charset() {
        Python::attach(|py| {
            let spec = TextBuilder::new(32).build().unwrap();
            let text = make_space(py, &spec).unwrap();

            assert!(
                text.call_method1("contains", ("pick up the object!",))
                    .unwrap()
                    .extract::<bool>()
                    .unwrap()
            );
        });
    }
}
