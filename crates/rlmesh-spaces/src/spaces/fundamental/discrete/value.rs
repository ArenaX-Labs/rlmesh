use crate::errors::{SpaceError, err_space};
use crate::spaces::{SpaceKind, SpaceSpec, SpaceValue};

pub(crate) fn contains_discrete(
    space: &SpaceSpec,
    value: &SpaceValue,
    path: &str,
) -> Result<(), SpaceError> {
    let val = match value {
        SpaceValue::Discrete(v) => *v,
        _ => return err_space!(path, "expected Discrete value"),
    };

    let d = match &space.spec {
        Some(SpaceKind::Discrete(d)) => d,
        _ => return err_space!(path, "space is not Discrete"),
    };

    // Check value is in range [start, start + n).
    // Compute `val - d.start` instead of `d.start + d.n` so a range ending at
    // i64::MAX (start + n - 1 == i64::MAX) does not overflow in containment.
    let in_range = val >= d.start && val.wrapping_sub(d.start) < d.n;
    if !in_range {
        // d.start + d.n may overflow; render the exclusive end safely.
        let end = match d.start.checked_add(d.n) {
            Some(end) => end.to_string(),
            None => "i64::MAX+1".to_string(),
        };
        return err_space!(
            path,
            format!("value {} not in range [{}, {})", val, d.start, end)
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::spaces::fundamental::DiscreteBuilder;
    use crate::spaces::{SpaceValue, contains};

    #[test]
    fn test_discrete_contains() {
        let space = DiscreteBuilder::new(4).build().unwrap();

        assert!(contains(&space, &SpaceValue::Discrete(0)).is_ok());
        assert!(contains(&space, &SpaceValue::Discrete(3)).is_ok());
        assert!(contains(&space, &SpaceValue::Discrete(4)).is_err());
        assert!(contains(&space, &SpaceValue::Discrete(-1)).is_err());
    }

    #[test]
    fn test_discrete_with_start() {
        let space = DiscreteBuilder::new(4).start(10).build().unwrap();

        assert!(contains(&space, &SpaceValue::Discrete(10)).is_ok());
        assert!(contains(&space, &SpaceValue::Discrete(13)).is_ok());
        assert!(contains(&space, &SpaceValue::Discrete(9)).is_err());
        assert!(contains(&space, &SpaceValue::Discrete(14)).is_err());
    }

    #[test]
    fn test_discrete_range_ending_at_i64_max() {
        // start + (n-1) == i64::MAX is a valid spec; containment must not overflow
        // when evaluating the exclusive upper bound (start + n).
        let space = DiscreteBuilder::new(4).start(i64::MAX - 3).build().unwrap();

        // All four values in [MAX-3, MAX] are valid.
        assert!(contains(&space, &SpaceValue::Discrete(i64::MAX - 3)).is_ok());
        assert!(contains(&space, &SpaceValue::Discrete(i64::MAX)).is_ok());
        // Just below the range.
        assert!(contains(&space, &SpaceValue::Discrete(i64::MAX - 4)).is_err());
    }
}
