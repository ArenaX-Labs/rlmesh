use std::time::Duration;

pub(super) fn average_ms(samples: &[Duration]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    Some(
        samples
            .iter()
            .map(|duration| duration.as_secs_f64() * 1000.0)
            .sum::<f64>()
            / samples.len() as f64,
    )
}

pub(super) fn percentile_ms(samples: &[Duration], percentile: f64) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }

    let mut values = samples
        .iter()
        .map(|duration| duration.as_secs_f64() * 1000.0)
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);

    let index = ((values.len() - 1) as f64 * percentile).ceil() as usize;
    values.get(index).copied()
}
