use std::collections::BTreeMap;
use std::fmt;

use super::value::{self, Value};
use crate::error::ApplyError;
use crate::path::{NodePath, PathSeg};
use crate::plans::{OBS_ROOT_KEY, envelope_key};

struct PathDisplay<'a>(&'a [PathSeg]);

impl fmt::Display for PathDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return f.write_str("<root>");
        }
        for (position, segment) in self.0.iter().enumerate() {
            match segment {
                PathSeg::Key(key) => {
                    if position > 0 {
                        f.write_str(".")?;
                    }
                    f.write_str(key)?;
                }
                PathSeg::Index(index) => write!(f, "[{index}]")?,
            }
        }
        Ok(())
    }
}

macro_rules! resolve_walk {
    (
        $root:expr, $path:expr,
        $map_get:ident, $list_get:ident,
        |$key:ident, $map:ident| $missing_key:expr,
        |$index:ident, $items:ident, $len:ident| $oob:expr,
        |$pseg:ident, $other:ident| $mismatch:expr $(,)?
    ) => {{
        let mut cur = $root;
        for segment in $path {
            cur = match (segment, cur) {
                (PathSeg::Key($key), Value::Map($map)) => {
                    $map.$map_get($key).ok_or_else(|| $missing_key)?
                }
                (PathSeg::Index($index), Value::List($items) | Value::Tuple($items)) => {
                    let $len = $items.len();
                    $items.$list_get(*$index).ok_or_else(|| $oob)?
                }
                ($pseg, $other) => return Err($mismatch),
            };
        }
        Ok(cur)
    }};
}

/// Resolve a [`NodePath`] source against the raw-obs *envelope*: the top-level
/// `BTreeMap` whose entries are keyed by the first path segment (a Dict-rooted
/// source) or by the reserved root key (an empty or Tuple-rooted source whose
/// whole `Value` is one envelope entry — see `envelope_key`), then walk the
/// remaining segments with [`resolve_source`].
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
    let remaining: &[PathSeg] = if key == OBS_ROOT_KEY {
        &source.0
    } else {
        source.rest()
    };
    resolve_source(entry, remaining)
}

/// Resolve a structured path into a (possibly nested) value tree.
///
/// An empty path returns the value itself, a `Key` step descends a
/// [`Value::Map`] and an `Index` step a [`Value::List`]; a shape that cannot
/// take the next step is an error, never a silent miss.
pub fn resolve_source<'obs>(
    root: &'obs Value,
    path: &[PathSeg],
) -> Result<&'obs Value, ApplyError> {
    resolve_walk!(
        root,
        path,
        get,
        get,
        |key, map| {
            let available: Vec<String> = map.keys().map(|key| format!("'{key}'")).collect();
            ApplyError::new(format!(
                "observation path '{}' is missing key '{key}'; available: [{}]",
                PathDisplay(path),
                available.join(", ")
            ))
        },
        |index, _items, len| {
            ApplyError::new(format!(
                "observation path '{}' index [{index}] is out of range (len {len})",
                PathDisplay(path)
            ))
        },
        |segment, other| {
            let (want, at) = match segment {
                PathSeg::Key(key) => ("a Dict", format!("'{key}'")),
                PathSeg::Index(index) => ("a Tuple", format!("[{index}]")),
            };
            ApplyError::new(format!(
                "observation path '{}' expected {want} at {at}, got {}",
                PathDisplay(path),
                value_kind(other)
            ))
        },
    )
}

/// Mutable variant of [`resolve_source`]: walk a [`NodePath`] into a `Value`
/// tree, returning a mutable reference to the addressed node (the whole tree for
/// the root/empty path).
pub fn resolve_source_mut<'obs>(
    root: &'obs mut Value,
    path: &NodePath,
) -> Result<&'obs mut Value, ApplyError> {
    let segments = &path.0[..];
    resolve_walk!(
        root,
        segments,
        get_mut,
        get_mut,
        |key, _map| ApplyError::new(format!("payload path '{path}' is missing key '{key}'")),
        |index, _items, len| {
            ApplyError::new(format!(
                "payload path '{path}' index [{index}] is out of range (len {len})"
            ))
        },
        |segment, other| {
            let (want, at) = match segment {
                PathSeg::Key(key) => ("a Dict", format!("'{key}'")),
                PathSeg::Index(index) => ("a Tuple", format!("[{index}]")),
            };
            ApplyError::new(format!(
                "payload path '{path}' expected {want} at {at}, got {}",
                value_kind(other)
            ))
        },
    )
}

fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::Tensor(_) => "a tensor",
        Value::Text(_) => "text",
        Value::Bytes(_) => "bytes",
        Value::None => "None",
        Value::Bool(_) => "a bool",
        Value::Int(_) => "an integer",
        Value::Tuple(_) => "a tuple",
        Value::Number(_) => "a number",
        Value::List(_) => "a list",
        Value::Map(_) => "a map",
    }
}

/// Return a flat float32 vector from a raw numeric value.
pub fn numeric_vector(value: &Value) -> Result<Vec<f32>, ApplyError> {
    match value {
        Value::None => Err(ApplyError::new("expected a numeric value, got None")),
        Value::Tensor(tensor) => Ok(value::to_f32_vec(tensor)),
        Value::Int(number) => Ok(vec![*number as f32]),
        Value::Bool(flag) => Ok(vec![f32::from(u8::from(*flag))]),
        Value::Number(number) => Ok(vec![*number as f32]),
        Value::List(items) | Value::Tuple(items) => {
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
