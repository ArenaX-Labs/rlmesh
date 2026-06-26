//! Observation payload access helpers.

use std::collections::BTreeMap;

use super::value::{self, Value};
use crate::error::ApplyError;
use crate::path::{NodePath, PathSeg};
use crate::plans::{OBS_ROOT_KEY, envelope_key};

/// Resolve a [`NodePath`] source against the raw-obs *envelope*: the top-level
/// `BTreeMap` whose entries are keyed by the first path segment (a Dict-rooted
/// source) or by the reserved root key (an empty or Tuple-rooted source whose
/// whole `Value` is one envelope entry — see `envelope_key`).
///
/// This is the apply-side successor to the old dotted-string `lookup`: it selects
/// the envelope entry, then walks the remaining segments via [`resolve_source`].
pub fn resolve_in_obs<'obs>(
    raw_obs: &'obs BTreeMap<String, Value>,
    source: &NodePath,
) -> Result<&'obs Value, ApplyError> {
    let key = envelope_key(source);
    let entry = raw_obs.get(&key).ok_or_else(|| {
        let available: Vec<String> = raw_obs.keys().map(|key| format!("'{key}'")).collect();
        ApplyError::new(format!(
            "observation is missing entry '{key}' for source '{source}'; available: [{}]",
            available.join(", ")
        ))
    })?;
    // The envelope entry is the value at the first segment (Dict-rooted) or the
    // whole obs (root/Tuple-rooted, under the reserved key). For a Dict-rooted
    // source the first segment has been consumed by the key selection, so walk
    // only the rest; otherwise walk the full path against the whole-obs value.
    let remaining: NodePath = if key == OBS_ROOT_KEY {
        source.clone()
    } else {
        NodePath(source.rest().to_vec())
    };
    resolve_source(entry, &remaining)
}

/// Resolve a structured [`NodePath`] into a (possibly nested) value tree.
///
/// An empty path returns the value itself (the root / single leaf), a `Key` step
/// descends a [`Value::Map`], and an `Index` step descends a [`Value::List`]. A
/// shape that cannot take the next step is a typed error rather than a silent
/// miss.
pub fn resolve_source<'obs>(root: &'obs Value, path: &NodePath) -> Result<&'obs Value, ApplyError> {
    let mut cur = root;
    for segment in &path.0 {
        cur = match (segment, cur) {
            (PathSeg::Key(key), Value::Map(map)) => map.get(key).ok_or_else(|| {
                let available: Vec<String> = map.keys().map(|key| format!("'{key}'")).collect();
                ApplyError::new(format!(
                    "observation path '{path}' is missing key '{key}'; available: [{}]",
                    available.join(", ")
                ))
            })?,
            (PathSeg::Index(index), Value::List(items)) => items.get(*index).ok_or_else(|| {
                ApplyError::new(format!(
                    "observation path '{path}' index [{index}] is out of range (len {})",
                    items.len()
                ))
            })?,
            (segment, other) => {
                let (want, at) = match segment {
                    PathSeg::Key(key) => ("a Dict", format!("'{key}'")),
                    PathSeg::Index(index) => ("a Tuple", format!("[{index}]")),
                };
                return Err(ApplyError::new(format!(
                    "observation path '{path}' expected {want} at {at}, got {}",
                    value_kind(other)
                )));
            }
        };
    }
    Ok(cur)
}

/// Mutable variant of [`resolve_source`]: walk a [`NodePath`] into a `Value`
/// tree, returning a mutable reference to the addressed node (the whole tree for
/// the root/empty path). Used by frame-stacking to read-and-replace a stacked
/// payload leaf in place.
pub fn resolve_source_mut<'obs>(
    root: &'obs mut Value,
    path: &NodePath,
) -> Result<&'obs mut Value, ApplyError> {
    let mut cur = root;
    for segment in &path.0 {
        cur = match (segment, cur) {
            (PathSeg::Key(key), Value::Map(map)) => map.get_mut(key).ok_or_else(|| {
                ApplyError::new(format!("payload path '{path}' is missing key '{key}'"))
            })?,
            (PathSeg::Index(index), Value::List(items)) => {
                let len = items.len();
                items.get_mut(*index).ok_or_else(|| {
                    ApplyError::new(format!(
                        "payload path '{path}' index [{index}] is out of range (len {len})"
                    ))
                })?
            }
            (segment, other) => {
                let (want, at) = match segment {
                    PathSeg::Key(key) => ("a Dict", format!("'{key}'")),
                    PathSeg::Index(index) => ("a Tuple", format!("[{index}]")),
                };
                return Err(ApplyError::new(format!(
                    "payload path '{path}' expected {want} at {at}, got {}",
                    value_kind(other)
                )));
            }
        };
    }
    Ok(cur)
}

/// A short human label for a [`Value`]'s kind, for path-mismatch errors.
fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::Tensor(_) => "a tensor",
        Value::Text(_) => "text",
        Value::Bytes(_) => "bytes",
        Value::Number(_) => "a number",
        Value::List(_) => "a list",
        Value::Map(_) => "a map",
    }
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
        Value::Bytes(_) => Err(ApplyError::new(
            "bytes values are only valid for image adapter inputs".to_owned(),
        )),
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
            "source range for an affine map has zero width".to_owned(),
        ));
    }
    for entry in value {
        *entry = (*entry - src_low) / span * (dst_high - dst_low) + dst_low;
    }
    Ok(())
}
