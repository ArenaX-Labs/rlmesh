use std::time::Duration;

pub(super) fn average_ms(samples: &[Duration]) -> Option<f64> {
    average_f64_iter(samples.iter().map(duration_ms), samples.len())
}

pub(super) fn percentile_ms(samples: &[Duration], percentile: f64) -> Option<f64> {
    let mut values = samples.iter().map(duration_ms).collect::<Vec<_>>();
    percentile_f64(&mut values, percentile)
}

pub(super) fn average_f64(samples: &[f64]) -> Option<f64> {
    average_f64_iter(samples.iter().copied(), samples.len())
}

pub(super) fn percentile_f64_samples(samples: &[f64], percentile: f64) -> Option<f64> {
    let mut values = samples.to_vec();
    percentile_f64(&mut values, percentile)
}

fn duration_ms(duration: &Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn average_f64_iter(values: impl Iterator<Item = f64>, len: usize) -> Option<f64> {
    if len == 0 {
        return None;
    }
    Some(values.sum::<f64>() / len as f64)
}

/// Nearest-rank percentile over an owned, sortable f64 buffer.
fn percentile_f64(values: &mut [f64], percentile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let index = ((values.len() - 1) as f64 * percentile).ceil() as usize;
    values.get(index).copied()
}
