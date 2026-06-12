use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};
use rlmesh_spaces::spaces::*;

pub fn make_text<'py>(
    py: Python<'py>,
    spaces: &Bound<'py, PyAny>,
    space: &SpaceSpec,
) -> PyResult<Bound<'py, PyAny>> {
    const PRINTABLE_ASCII: &str = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\u{0b}\u{0c}";
    let text_spec = match &space.spec {
        Some(space_spec::Spec::Text(spec)) => spec,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err("spec.text missing"));
        }
    };

    let kwargs = PyDict::new(py);
    kwargs.set_item("min_length", text_spec.min_length)?;
    kwargs.set_item("max_length", text_spec.max_length)?;
    kwargs.set_item(
        "charset",
        if text_spec.charset.is_empty() {
            PRINTABLE_ASCII
        } else {
            text_spec.charset.as_str()
        },
    )?;

    spaces.getattr("Text")?.call((), Some(&kwargs))
}

pub fn parse_text<'py>(space: &Bound<'py, PyAny>) -> PyResult<SpaceSpec> {
    let max_length = space.getattr("max_length")?.extract::<i64>()?;
    let min_length = match space.getattr("min_length") {
        Ok(value) => value.extract::<i64>().unwrap_or(1),
        Err(_) => 1,
    };
    let charset = rlmesh_text_charset(text_charset(space)?);

    let mut builder = TextBuilder::new(max_length).min_length(min_length);
    if let Some(charset) = charset {
        builder = builder.charset(charset);
    }
    builder
        .build()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

fn text_charset(space: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(value) = space.getattr("characters")
        && let Ok(characters) = value.extract::<String>()
    {
        return Ok(characters);
    }

    if let Ok(value) = space.getattr("character_list")
        && let Some(characters) = iterable_text_charset(&value, false)?
    {
        return Ok(characters);
    }

    if let Ok(value) = space.getattr("character_set")
        && let Some(characters) = iterable_text_charset(&value, true)?
    {
        return Ok(characters);
    }

    if let Ok(value) = space.getattr("charset") {
        if value.is_none() {
            return Ok(String::new());
        }
        if let Ok(characters) = value.extract::<String>() {
            return Ok(characters);
        }
        if let Some(characters) = iterable_text_charset(&value, true)? {
            return Ok(characters);
        }
    }

    Ok(String::new())
}

fn iterable_text_charset(value: &Bound<'_, PyAny>, sort: bool) -> PyResult<Option<String>> {
    let Ok(iter) = value.try_iter() else {
        return Ok(None);
    };
    let mut characters = iter
        .map(|item| item.and_then(|item| item.extract::<String>()))
        .collect::<PyResult<Vec<_>>>()?;
    if sort {
        characters.sort();
    }
    Ok(Some(characters.join("")))
}

fn rlmesh_text_charset(charset: String) -> Option<String> {
    // Gymnasium's default Text charset is alphanumeric-only; RLMesh treats it as generic text.
    if charset.is_empty() || is_gymnasium_default_text_charset(&charset) {
        None
    } else {
        Some(charset)
    }
}

fn is_gymnasium_default_text_charset(charset: &str) -> bool {
    const DEFAULT: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    charset.chars().count() == DEFAULT.chars().count()
        && DEFAULT.chars().all(|character| charset.contains(character))
}
