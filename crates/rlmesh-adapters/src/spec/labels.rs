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
