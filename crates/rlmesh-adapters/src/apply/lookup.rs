//! Observation payload access helpers.

use std::collections::BTreeMap;
use std::fmt;

use super::value::{self, Value};
use crate::error::ApplyError;
use crate::path::{NodePath, PathSeg};
use crate::plans::{OBS_ROOT_KEY, envelope_key};

/// Render a borrowed `[PathSeg]` slice exactly like [`NodePath`]'s `Display`
/// (`<root>` for empty, dotted `Key`s, bracketed `Index`es) without allocating a
/// `NodePath` to wrap it — so the resolve walk can take a slice (no per-call
/// clone of the source path) yet keep identical path text in its error messages.
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

/// Walk a structured path into a `Value` tree, shared by the shared-ref and
/// mutable-ref resolvers. The two differ only in the borrow (`get` vs `get_mut`)
/// and their error wording, so this macro is the single source of the descent
/// logic: each invocation supplies the map/list accessor and the error
/// expressions, keeping the deliberately distinct `observation`/`payload`
/// messages intact while the navigation cannot drift between them.
///
/// `$path` is a `&[PathSeg]` slice so the immutable resolver can be handed the
/// source's `rest()` (or whole) segments without cloning a `NodePath`. The
/// out-of-range error expression receives the list `len`, captured before the
/// element borrow so the mutable `get_mut` arm still type-checks.
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
                (PathSeg::Index($index), Value::List($items)) => {
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
    // Both are borrowed slices of `source` -- no `NodePath` clone per step.
    let remaining: &[PathSeg] = if key == OBS_ROOT_KEY {
        &source.0
    } else {
        source.rest()
    };
    resolve_source(entry, remaining)
}

/// Resolve a structured path into a (possibly nested) value tree.
///
/// An empty path returns the value itself (the root / single leaf), a `Key` step
/// descends a [`Value::Map`], and an `Index` step descends a [`Value::List`]. A
/// shape that cannot take the next step is a typed error rather than a silent
/// miss. The path is a borrowed `[PathSeg]` slice so callers can pass a sub-path
/// of a [`NodePath`] without cloning it.
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
/// the root/empty path). Used by frame-stacking to read-and-replace a stacked
/// payload leaf in place.
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
