//! The `labels` attribute: per-axis names on a numeric leaf.
//!
//! A label tuple names the axes of a joint vector (`FR_hip_joint`, `FR_thigh_joint`, ...)
//! so two sides that list the same names in different orders resolve to a
//! permutation instead of a silent misalignment. Labels are plain strings on
//! the wire; both sides must name the same set.

/// The codec rules every labeled leaf shares: at least one label, none
/// repeated. Width agreement is checked where the width is known (the codec
/// for a `Field`/`Actuator`, `join` for a `StateTag`, resolve for a part).
pub(crate) fn check_labels(labels: &[String], locus: &str) -> Result<(), String> {
    if labels.is_empty() {
        return Err(format!("{locus}: labels must name at least one axis"));
    }
    let mut seen = std::collections::BTreeSet::new();
    for label in labels {
        if !seen.insert(label.as_str()) {
            return Err(format!("{locus}: label {label:?} is repeated"));
        }
    }
    Ok(())
}

/// A per-axis vector on the wire must be finite in every entry (the scalar
/// forms are finite by serde_json's own rejection of `Infinity`/`NaN`; a
/// vector element overflowing to infinity is rejected the same way, so this
/// pins the invariant rather than adding one).
pub(crate) fn check_axis(values: Option<&[f64]>, name: &str, locus: &str) -> Result<(), String> {
    if let Some(values) = values {
        if values.is_empty() {
            return Err(format!("{locus}: {name} must carry at least one value"));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(format!("{locus}: {name} must be finite"));
        }
    }
    Ok(())
}
