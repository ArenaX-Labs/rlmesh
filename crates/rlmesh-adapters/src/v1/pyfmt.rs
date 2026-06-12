//! Python-`repr()` style formatting shared by resolver errors and
//! `describe()` output.
//!
//! These strings are part of the cross-implementation contract (the
//! conformance vectors pin them), so every formatter here must match the
//! reference implementation byte for byte. Roles and keys never contain
//! quotes, so the single-quote form is always correct.

use std::collections::BTreeMap;

use super::spec::RotationEncoding;

/// Python-style `repr()` of a string.
pub(crate) fn py_repr(value: &str) -> String {
    format!("'{value}'")
}

/// Python-style `repr()` of an optional rotation encoding.
pub(crate) fn py_repr_encoding(value: Option<RotationEncoding>) -> String {
    match value {
        Some(encoding) => format!("'{}'", encoding.as_str()),
        None => "None".to_owned(),
    }
}

/// Python-style `repr()` of a float pair, e.g. `(-1.0, 1.0)`.
pub(crate) fn py_repr_range(range: (f64, f64)) -> String {
    format!("({:?}, {:?})", range.0, range.1)
}

/// Python-style `repr()` of a map's sorted keys, e.g. `['a', 'b']`.
pub(crate) fn py_repr_sorted_keys<V>(map: &BTreeMap<String, V>) -> String {
    let items: Vec<String> = map.keys().map(|key| py_repr(key)).collect();
    format!("[{}]", items.join(", "))
}
