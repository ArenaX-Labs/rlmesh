//! Validate the C/C++ host runtime declaration for served describe envelopes.
use crate::abi::status::CapiError;
use serde_json::{Value, json};

/// Validate the optional host declaration before it can override runtime fields.
/// Component, package, OS/arch and protocol/edition remain library-owned.
pub(crate) fn host_runtime(text: &str) -> Result<Value, CapiError> {
    let invalid = |message: String| CapiError::invalid_arg(format!("host_runtime_json: {message}"));
    let value: Value = serde_json::from_str(text).map_err(|err| invalid(err.to_string()))?;
    let object = value
        .as_object()
        .ok_or_else(|| invalid("expected a JSON object".into()))?;
    let non_empty = |value: &Value, field: &str| {
        value
            .as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .ok_or_else(|| invalid(format!("{field} must be a non-empty string")))
    };
    let mut normalized = serde_json::Map::new();
    for (key, value) in object {
        let value = match key.as_str() {
            "language" | "language_version" => Value::String(non_empty(value, key)?),
            "framework_versions" => {
                let versions = value
                    .as_object()
                    .ok_or_else(|| invalid("framework_versions must be an object".into()))?;
                let mut normalized_versions = serde_json::Map::new();
                for (name, version) in versions {
                    let name = name.trim();
                    if name.is_empty() {
                        return Err(invalid("framework names must be non-empty".into()));
                    }
                    let version = non_empty(version, "framework version")?;
                    if normalized_versions
                        .insert(name.to_owned(), Value::String(version))
                        .is_some()
                    {
                        return Err(invalid(format!(
                            "duplicate framework name {name:?} after trimming"
                        )));
                    }
                }
                Value::Object(normalized_versions)
            }
            "toolchain" => {
                let toolchain = value
                    .as_object()
                    .ok_or_else(|| invalid("toolchain must be an object".into()))?;
                if toolchain.len() != 2
                    || !toolchain.contains_key("name")
                    || !toolchain.contains_key("version")
                {
                    return Err(invalid(
                        "toolchain must contain only name and version".into(),
                    ));
                }
                let name = non_empty(&toolchain["name"], "toolchain.name")?;
                let version = non_empty(&toolchain["version"], "toolchain.version")?;
                if !matches!(name.as_str(), "gcc" | "clang" | "msvc") {
                    return Err(invalid("toolchain.name must be gcc, clang or msvc".into()));
                }
                if !version.split_once('.').is_some_and(|(major, minor)| {
                    !major.is_empty()
                        && !minor.is_empty()
                        && major.bytes().all(|b| b.is_ascii_digit())
                        && minor.bytes().all(|b| b.is_ascii_digit())
                }) {
                    return Err(invalid("toolchain.version must be major.minor".into()));
                }
                json!({"name": name, "version": version})
            }
            _ => return Err(invalid(format!("unknown field {key:?}"))),
        };
        normalized.insert(key.clone(), value);
    }
    Ok(Value::Object(normalized))
}
