//! Observation payload access helpers.

use std::collections::BTreeMap;

use super::error::ApplyError;
use super::value::{self, Value};

/// Fetch a value from an observation map, traversing dotted paths.
pub fn lookup<'obs>(
    raw_obs: &'obs BTreeMap<String, Value>,
    key: &str,
) -> Result<&'obs Value, ApplyError> {
    if let Some(value) = raw_obs.get(key) {
        return Ok(value);
    }
    let missing = || {
        let available: Vec<String> = raw_obs.keys().map(|key| format!("'{key}'")).collect();
        ApplyError::new(format!(
            "observation is missing key '{key}'; available keys: [{}]",
            available.join(", ")
        ))
    };
    let mut segments = key.split('.');
    let first = segments.next().ok_or_else(missing)?;
    let mut value = raw_obs.get(first).ok_or_else(missing)?;
    for segment in segments {
        let Value::Map(map) = value else {
            return Err(missing());
        };
        value = map.get(segment).ok_or_else(missing)?;
    }
    Ok(value)
}

/// Return a flat float32 vector from a raw numeric value.
pub fn numeric_vector(value: &Value) -> Result<Vec<f32>, ApplyError> {
    match value {
        Value::Tensor(tensor) => Ok(value::to_f32_vec(tensor)),
        Value::Number(number) => Ok(vec![*number as f32]),
        Value::List(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.extend(numeric_vector(item)?);
            }
            Ok(out)
        }
        Value::Map(map) => {
            let Some(data) = map.get("data") else {
                let keys: Vec<String> = map.keys().map(|key| format!("'{key}'")).collect();
                return Err(ApplyError::new(format!(
                    "expected numeric payload mapping with a 'data' key, got \
                     keys [{}]",
                    keys.join(", ")
                )));
            };
            numeric_vector(data)
        }
        Value::Text(_) => Err(ApplyError::new(
            "expected a numeric value, got text".to_owned(),
        )),
    }
}

/// Affinely map values from the `src` range into the `dst` range.
pub fn map_range(value: &mut [f32], src: (f64, f64), dst: (f64, f64)) -> Result<(), ApplyError> {
    let (src_low, src_high) = (src.0 as f32, src.1 as f32);
    let (dst_low, dst_high) = (dst.0 as f32, dst.1 as f32);
    let span = src_high - src_low;
    if span == 0.0 {
        return Err(ApplyError::new(
            "source range for action mapping has zero width".to_owned(),
        ));
    }
    for entry in value {
        *entry = (*entry - src_low) / span * (dst_high - dst_low) + dst_low;
    }
    Ok(())
}
