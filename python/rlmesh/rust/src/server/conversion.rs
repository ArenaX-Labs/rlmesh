//! Conversions between gymnasium Python objects and RLMesh native values:
//! reset/step result normalization, render-frame PNG encoding, and env metadata
//! extraction (render mode, autoreset convention).

use image::ColorType;
use image::ImageEncoder;
use image::codecs::png::PngEncoder;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use rlmesh_spaces::spaces::SpaceSpec;
use rlmesh_spaces::{AutoresetMode, MetaMap};

use crate::spaces::{ValueBackend, py_any_to_meta_map, py_any_to_space_value_with_backend};

pub(super) fn extract_optional_meta_attr(
    obj: &Bound<'_, PyAny>,
    attr_name: &str,
) -> PyResult<Option<MetaMap>> {
    if !obj.hasattr(attr_name)? {
        return Ok(None);
    }

    let value = obj.getattr(attr_name)?;
    if value.is_none() {
        return Ok(None);
    }

    Ok(Some(py_any_to_meta_map(&value)?))
}

pub(super) fn extract_render_mode(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    if !obj.hasattr("render_mode")? {
        return Ok(String::new());
    }

    let value = obj.getattr("render_mode")?;
    if value.is_none() {
        return Ok(String::new());
    }

    value.extract::<String>()
}

/// Derive the per-lane autoreset convention from the env's
/// `metadata["autoreset_mode"]` (gymnasium vector envs always set it). The value
/// is either a gymnasium `AutoresetMode` enum (whose `.value` is one of
/// `"NextStep"`/`"SameStep"`/`"Disabled"`) or that plain string.
///
/// An *absent* key (or no metadata at all) defaults to `Disabled` — a
/// scalar/custom env naturally needs explicit reset. A *present* value is held
/// to the contract: an explicit `None`, a non-string, or an unrecognized string
/// raises `ValueError` rather than silently downgrading to `Disabled`, which
/// would double-reset a mislabeled self-autoresetting env. `SameStep` is reserved
/// in the protocol but not yet honored by the runtime, so it too is rejected
/// (fail loud rather than mishandle timing).
pub(super) fn derive_autoreset_mode(obj: &Bound<'_, PyAny>) -> PyResult<AutoresetMode> {
    if !obj.hasattr("metadata")? {
        return Ok(AutoresetMode::Disabled);
    }
    let metadata = obj.getattr("metadata")?;
    if metadata.is_none() {
        return Ok(AutoresetMode::Disabled);
    }
    // A missing key (or a non-subscriptable metadata) defaults to Disabled — a
    // scalar/custom env legitimately omits it. But a key that is *present* and
    // explicitly `None` is a deliberate value held to the contract, not the same
    // as "absent": fail loud rather than silently downgrade to Disabled.
    let mode_obj = match metadata.get_item("autoreset_mode") {
        Err(_) => return Ok(AutoresetMode::Disabled),
        Ok(value) if value.is_none() => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "metadata[\"autoreset_mode\"] is present but None; set it to a gymnasium \
                 AutoresetMode enum or one of \"NextStep\"/\"SameStep\"/\"Disabled\", or omit the \
                 key entirely to default to DISABLED",
            ));
        }
        Ok(value) => value,
    };

    // A gymnasium AutoresetMode enum carries the canonical string in `.value`;
    // a wire-degraded value is already a plain string. The key is present (past
    // the get_item guard), so a non-string value is a contract violation: fail
    // loud rather than collapse to "" and silently downgrade to DISABLED.
    let non_string_err = || {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "metadata[\"autoreset_mode\"] must be a gymnasium AutoresetMode enum or one of \
             \"NextStep\"/\"SameStep\"/\"Disabled\", got {mode_obj:?}"
        ))
    };
    let mode_str: String = if mode_obj.hasattr("value")? {
        mode_obj
            .getattr("value")?
            .extract()
            .map_err(|_| non_string_err())?
    } else {
        mode_obj.extract().map_err(|_| non_string_err())?
    };

    match mode_str.as_str() {
        "NextStep" => Ok(AutoresetMode::NextStep),
        "Disabled" => Ok(AutoresetMode::Disabled),
        "SameStep" => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "SAME_STEP autoreset mode is not yet supported by the rlmesh runtime; \
             construct the environment with NEXT_STEP or DISABLED autoreset",
        )),
        // Present but an unrecognized string — a typo or a mode this runtime
        // does not know. Erroring beats silently downgrading to DISABLED, which
        // would double-reset a self-autoresetting env.
        other => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "unrecognized metadata[\"autoreset_mode\"] {other:?}; expected a gymnasium \
             AutoresetMode enum or one of \"NextStep\"/\"SameStep\"/\"Disabled\""
        ))),
    }
}

pub(super) type SingleStepResultParts<'py> =
    (Bound<'py, PyAny>, f64, bool, bool, Bound<'py, PyAny>);
pub(super) type VectorStepResultParts<'py> = (
    Bound<'py, PyAny>,
    Vec<f64>,
    Vec<bool>,
    Vec<bool>,
    Bound<'py, PyAny>,
);

pub(super) fn normalize_reset_result<'py>(
    py: Python<'py>,
    result: Bound<'py, PyAny>,
    observation_space: &SpaceSpec,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    if let Ok(tuple) = result.cast::<PyTuple>()
        && tuple.len() == 2
    {
        let info = tuple.get_item(1)?;
        if info.is_none() || info.cast::<PyDict>().is_ok() {
            let obs = tuple.get_item(0)?;
            if !is_tuple_space(observation_space)
                || py_any_to_space_value_with_backend(
                    py,
                    &obs,
                    observation_space,
                    ValueBackend::Auto,
                )
                .is_ok()
            {
                return Ok((obs, info));
            }
        }
    }

    Ok((result, PyDict::new(py).into_any()))
}

pub(super) fn is_tuple_space(space: &SpaceSpec) -> bool {
    matches!(
        space.spec.as_ref(),
        Some(rlmesh_spaces::spaces::SpaceKind::Tuple(_))
    )
}

pub(super) fn normalize_single_step_result<'py>(
    result: Bound<'py, PyAny>,
) -> PyResult<SingleStepResultParts<'py>> {
    let tuple = result.cast::<PyTuple>()?;
    match tuple.len() {
        5 => Ok((
            tuple.get_item(0)?,
            tuple.get_item(1)?.extract::<f64>()?,
            tuple.get_item(2)?.extract::<bool>()?,
            tuple.get_item(3)?.extract::<bool>()?,
            tuple.get_item(4)?,
        )),
        4 => {
            let done = tuple.get_item(2)?.extract::<bool>()?;
            let info = tuple.get_item(3)?;
            let truncated = done && legacy_single_truncated(&info)?;
            Ok((
                tuple.get_item(0)?,
                tuple.get_item(1)?.extract::<f64>()?,
                done && !truncated,
                truncated,
                info,
            ))
        }
        len => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "env.step() must return 4 legacy Gym values or 5 Gymnasium values, got {len}"
        ))),
    }
}

pub(super) fn normalize_vector_step_result<'py>(
    result: Bound<'py, PyAny>,
    num_envs: usize,
) -> PyResult<VectorStepResultParts<'py>> {
    let tuple = result.cast::<PyTuple>()?;
    match tuple.len() {
        5 => Ok((
            tuple.get_item(0)?,
            tuple.get_item(1)?.extract::<Vec<f64>>()?,
            tuple.get_item(2)?.extract::<Vec<bool>>()?,
            tuple.get_item(3)?.extract::<Vec<bool>>()?,
            tuple.get_item(4)?,
        )),
        4 => {
            let done = tuple.get_item(2)?.extract::<Vec<bool>>()?;
            let info = tuple.get_item(3)?;
            let truncated = legacy_vector_truncated(&info, num_envs)?;
            let terminated = done
                .iter()
                .zip(truncated.iter())
                .map(|(done, truncated)| *done && !*truncated)
                .collect();
            Ok((
                tuple.get_item(0)?,
                tuple.get_item(1)?.extract::<Vec<f64>>()?,
                terminated,
                truncated,
                info,
            ))
        }
        len => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "env.step() must return 4 legacy Gym values or 5 Gymnasium values, got {len}"
        ))),
    }
}

pub(super) fn legacy_single_truncated(info: &Bound<'_, PyAny>) -> PyResult<bool> {
    if let Ok(dict) = info.cast::<PyDict>()
        && let Some(value) = dict.get_item("TimeLimit.truncated")?
    {
        return value.extract::<bool>();
    }
    Ok(false)
}

pub(super) fn legacy_vector_truncated(
    info: &Bound<'_, PyAny>,
    num_envs: usize,
) -> PyResult<Vec<bool>> {
    if let Ok(dict) = info.cast::<PyDict>()
        && let Some(value) = dict.get_item("TimeLimit.truncated")?
    {
        if let Ok(values) = value.extract::<Vec<bool>>() {
            return Ok(values);
        }
        if value.hasattr("tolist")? {
            return value.call_method0("tolist")?.extract::<Vec<bool>>();
        }
        if let Ok(value) = value.extract::<bool>() {
            return Ok(vec![value; num_envs]);
        }
    }
    Ok(vec![false; num_envs])
}

pub(super) fn encode_render_png(
    py: Python<'_>,
    value: &Bound<'_, PyAny>,
) -> PyResult<Option<Vec<u8>>> {
    if value.is_none() {
        return Ok(None);
    }

    let numpy = py.import("numpy")?;
    let mut array = if value.hasattr("tobytes")? && value.hasattr("shape")? {
        value.clone()
    } else {
        numpy.call_method1("array", (value,))?
    };

    array = array.call_method1("astype", ("uint8",))?;
    let shape = array.getattr("shape")?.extract::<Vec<usize>>()?;
    let bytes = array.call_method0("tobytes")?.extract::<Vec<u8>>()?;

    let (width, height, color_type, data) = match shape.as_slice() {
        [height, width, 3] => (*width as u32, *height as u32, ColorType::Rgb8, bytes),
        [height, width, 4] => (*width as u32, *height as u32, ColorType::Rgba8, bytes),
        [3, height, width] => (
            *width as u32,
            *height as u32,
            ColorType::Rgb8,
            chw_to_hwc(bytes, 3, *height, *width),
        ),
        [4, height, width] => (
            *width as u32,
            *height as u32,
            ColorType::Rgba8,
            chw_to_hwc(bytes, 4, *height, *width),
        ),
        [height, width] => (*width as u32, *height as u32, ColorType::L8, bytes),
        _ => return Ok(None),
    };

    let mut encoded = Vec::new();
    PngEncoder::new(&mut encoded)
        .write_image(&data, width, height, color_type.into())
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.to_string()))?;
    Ok(Some(encoded))
}

pub(super) fn chw_to_hwc(bytes: Vec<u8>, channels: usize, height: usize, width: usize) -> Vec<u8> {
    let plane = height * width;
    let mut out = Vec::with_capacity(bytes.len());
    for index in 0..plane {
        for channel in 0..channels {
            out.push(bytes[channel * plane + index]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlmesh_spaces::spaces::{DiscreteBuilder, TupleSpaceBuilder};

    fn discrete_space() -> SpaceSpec {
        DiscreteBuilder::new(8).build().unwrap()
    }

    fn tuple_two_space() -> SpaceSpec {
        TupleSpaceBuilder::new()
            .with(DiscreteBuilder::new(8).build().unwrap())
            .with(DiscreteBuilder::new(8).build().unwrap())
            .build()
            .unwrap()
    }

    #[test]
    fn legacy_reset_obs_only_gets_empty_info() {
        Python::attach(|py| {
            let result = 7i64.into_pyobject(py).unwrap().into_any();
            let (obs, info) = normalize_reset_result(py, result, &discrete_space()).unwrap();

            assert_eq!(obs.extract::<i64>().unwrap(), 7);
            assert!(info.cast::<PyDict>().unwrap().is_empty());
        });
    }

    #[test]
    fn modern_reset_tuple_preserves_info() {
        Python::attach(|py| {
            let info = PyDict::new(py);
            info.set_item("seed", 123).unwrap();
            let result = (7i64, info).into_pyobject(py).unwrap().into_any();
            let (obs, info) = normalize_reset_result(py, result, &discrete_space()).unwrap();

            assert_eq!(obs.extract::<i64>().unwrap(), 7);
            assert_eq!(
                info.cast::<PyDict>()
                    .unwrap()
                    .get_item("seed")
                    .unwrap()
                    .unwrap()
                    .extract::<i64>()
                    .unwrap(),
                123
            );
        });
    }

    #[test]
    fn modern_reset_splits_two_tuple_observation_space() {
        Python::attach(|py| {
            let info = PyDict::new(py);
            info.set_item("seed", 123).unwrap();
            let result = ((7i64, 3i64), info).into_pyobject(py).unwrap().into_any();
            let (obs, info) = normalize_reset_result(py, result, &tuple_two_space()).unwrap();

            let obs_tuple = obs.cast::<PyTuple>().unwrap();
            assert_eq!(obs_tuple.len(), 2);
            assert_eq!(obs_tuple.get_item(0).unwrap().extract::<i64>().unwrap(), 7);
            assert_eq!(obs_tuple.get_item(1).unwrap().extract::<i64>().unwrap(), 3);
            assert_eq!(
                info.cast::<PyDict>()
                    .unwrap()
                    .get_item("seed")
                    .unwrap()
                    .unwrap()
                    .extract::<i64>()
                    .unwrap(),
                123
            );
        });
    }

    #[test]
    fn legacy_obs_only_two_tuple_is_not_split_for_tuple_space() {
        Python::attach(|py| {
            let second = PyDict::new(py);
            second.set_item("pos", 1).unwrap();
            let result = (7i64, second).into_pyobject(py).unwrap().into_any();
            let (obs, info) = normalize_reset_result(py, result, &tuple_two_space()).unwrap();

            let obs_tuple = obs.cast::<PyTuple>().unwrap();
            assert_eq!(obs_tuple.len(), 2);
            assert_eq!(obs_tuple.get_item(0).unwrap().extract::<i64>().unwrap(), 7);
            assert!(info.cast::<PyDict>().unwrap().is_empty());
        });
    }

    #[test]
    fn legacy_single_step_done_can_map_to_truncated() {
        Python::attach(|py| {
            let info = PyDict::new(py);
            info.set_item("TimeLimit.truncated", true).unwrap();
            let result = (1i64, 1.0f64, true, info)
                .into_pyobject(py)
                .unwrap()
                .into_any();
            let (obs, reward, terminated, truncated, _info) =
                normalize_single_step_result(result).unwrap();

            assert_eq!(obs.extract::<i64>().unwrap(), 1);
            assert_eq!(reward, 1.0);
            assert!(!terminated);
            assert!(truncated);
        });
    }

    #[test]
    fn legacy_vector_step_done_maps_each_time_limit_flag() {
        Python::attach(|py| {
            let info = PyDict::new(py);
            info.set_item("TimeLimit.truncated", vec![true, false])
                .unwrap();
            let result = (vec![1i64, 1], vec![1.0f64, 2.0], vec![true, true], info)
                .into_pyobject(py)
                .unwrap()
                .into_any();
            let (_obs, rewards, terminated, truncated, _info) =
                normalize_vector_step_result(result, 2).unwrap();

            assert_eq!(rewards, vec![1.0, 2.0]);
            assert_eq!(terminated, vec![false, true]);
            assert_eq!(truncated, vec![true, false]);
        });
    }

    /// Build a gymnasium-like env whose `metadata["autoreset_mode"]` is `value`
    /// (or omit the key entirely when `value` is None).
    fn env_with_mode<'py>(py: Python<'py>, value: Option<Bound<'py, PyAny>>) -> Bound<'py, PyAny> {
        let metadata = PyDict::new(py);
        if let Some(v) = value {
            metadata.set_item("autoreset_mode", v).unwrap();
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("metadata", metadata).unwrap();
        py.import("types")
            .unwrap()
            .getattr("SimpleNamespace")
            .unwrap()
            .call((), Some(&kwargs))
            .unwrap()
    }

    /// A gymnasium `AutoresetMode` enum stand-in: an object exposing `.value`.
    fn enum_like<'py>(py: Python<'py>, value: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("value", value).unwrap();
        py.import("types")
            .unwrap()
            .getattr("SimpleNamespace")
            .unwrap()
            .call((), Some(&kwargs))
            .unwrap()
    }

    fn pystr<'py>(py: Python<'py>, s: &str) -> Bound<'py, PyAny> {
        s.into_pyobject(py).unwrap().into_any()
    }

    #[test]
    fn derive_autoreset_mode_table() {
        Python::attach(|py| {
            // (env, expected): Some(mode) is accepted; None means it must error.
            let cases: Vec<(Bound<'_, PyAny>, Option<AutoresetMode>)> = vec![
                // gymnasium enum carries the canonical string in `.value`
                (
                    env_with_mode(py, Some(enum_like(py, pystr(py, "NextStep")))),
                    Some(AutoresetMode::NextStep),
                ),
                // wire-degraded plain string
                (
                    env_with_mode(py, Some(pystr(py, "Disabled"))),
                    Some(AutoresetMode::Disabled),
                ),
                // metadata present, no autoreset_mode key -> default Disabled
                (env_with_mode(py, None), Some(AutoresetMode::Disabled)),
                // no metadata attribute at all (scalar/custom env) -> Disabled
                (
                    7i64.into_pyobject(py).unwrap().into_any(),
                    Some(AutoresetMode::Disabled),
                ),
                // key present but explicitly None -> error, not a downgrade
                (env_with_mode(py, Some(py.None().into_bound(py))), None),
                // present non-string -> error
                (
                    env_with_mode(py, Some(1i64.into_pyobject(py).unwrap().into_any())),
                    None,
                ),
                // enum-like whose `.value` is non-string -> error
                (
                    env_with_mode(
                        py,
                        Some(enum_like(py, 1i64.into_pyobject(py).unwrap().into_any())),
                    ),
                    None,
                ),
                // unrecognized string -> error
                (env_with_mode(py, Some(pystr(py, "Bogus"))), None),
                // SAME_STEP is reserved but unsupported -> error
                (env_with_mode(py, Some(pystr(py, "SameStep"))), None),
            ];

            for (env, expected) in cases {
                let actual = derive_autoreset_mode(&env);
                match expected {
                    Some(mode) => assert_eq!(actual.unwrap(), mode),
                    None => assert!(
                        actual.is_err(),
                        "expected an error, got {actual:?} for {env:?}"
                    ),
                }
            }
        });
    }
}
