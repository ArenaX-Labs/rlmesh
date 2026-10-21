//! The element conversions v1 adapters, v2 bindings and the edge share
//! (rotations and images have their own modules).

/// `values[i]` broadcast: none is `default`, one applies to every element.
fn pick(values: &[f64], i: usize, default: f64) -> f64 {
    match values {
        [] => default,
        [one] => *one,
        many => many.get(i).copied().unwrap_or(default),
    }
}

/// The declared affine on element `i`: `x * scale + offset`, inputs and
/// actions alike. `scale` and `offset` are each empty (1 and 0), one value,
/// or one per element.
pub fn affine(x: f64, i: usize, scale: &[f64], offset: &[f64]) -> f64 {
    x * pick(scale, i, 1.0) + pick(offset, i, 0.0)
}

/// A binary command snapped to an end of its `[low, high]` range: `high`
/// from the midpoint up, `low` below it (v1 grippers: ±1 at 0; a v2 binary
/// command: 0/1 at 0.5).
pub fn binary(value: f64, low: f64, high: f64) -> f64 {
    if value >= (low + high) / 2.0 {
        high
    } else {
        low
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_affine_scales_then_offsets_with_broadcast() {
        assert_eq!(affine(0.3, 0, &[-2.0], &[1.0]), 0.3 * -2.0 + 1.0);
        assert_eq!(affine(2.0, 1, &[1.0, 3.0], &[]), 6.0);
        assert_eq!(affine(2.0, 5, &[], &[0.5]), 2.5);
    }

    #[test]
    fn binary_snaps_at_the_midpoint_and_the_midpoint_goes_high() {
        assert_eq!(binary(0.0, -1.0, 1.0), 1.0);
        assert_eq!(binary(-1e-9, -1.0, 1.0), -1.0);
        assert_eq!(binary(0.5, 0.0, 1.0), 1.0);
        assert_eq!(binary(0.49, 0.0, 1.0), 0.0);
    }
}
