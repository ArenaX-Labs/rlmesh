//! Human-readable formatting shared by resolver errors and `describe()`.
//!
//! These strings are *reference snapshots*: the conformance vectors pin them
//! so every implementation renders the same text, but they are NOT a stable
//! cross-language contract. Structural callers should match on typed errors
//! (e.g. [`JoinError`](super::JoinError)) rather than parse this text, and a
//! C++ or other binding is free to render its own wording.

use std::collections::BTreeMap;

use super::spec::RotationEncoding;

/// A quoted string, for keys and roles in messages.
pub(crate) fn quoted(value: &str) -> String {
    format!("{value:?}")
}

/// A quoted optional rotation encoding, or `None`.
pub(crate) fn quoted_encoding(value: Option<RotationEncoding>) -> String {
    match value {
        Some(encoding) => format!("{:?}", encoding.as_str()),
        None => "None".to_owned(),
    }
}

/// A float pair, e.g. `(-1.0, 1.0)`.
pub(crate) fn quoted_range(range: (f64, f64)) -> String {
    format!("{range:?}")
}

/// A map's sorted keys, quoted, e.g. `["a", "b"]`.
pub(crate) fn quoted_keys<V>(map: &BTreeMap<String, V>) -> String {
    let keys: Vec<&String> = map.keys().collect();
    format!("{keys:?}")
}
