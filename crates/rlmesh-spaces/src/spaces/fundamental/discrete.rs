use crate::errors::{SpaceError, err_space};
use crate::spaces::{SpaceKind, SpaceSpec, SpaceValue};
use crate::{DType, DiscreteSpec};

#[must_use = "a space builder does nothing until .build() is called"]
pub struct DiscreteBuilder {
    n: i64,
    start: i64,
    dtype: DType,
}

impl DiscreteBuilder {
    pub fn new(n: i64) -> Self {
        Self {
            n,
            start: 0,
            dtype: DType::Int64,
        }
    }
    pub fn start(mut self, start: i64) -> Self {
        self.start = start;
        self
    }
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }
    pub fn build(self) -> Result<SpaceSpec, SpaceError> {
        make_discrete_at(self.n, self.start, self.dtype)
    }
}

fn make_discrete_at(n: i64, start: i64, dtype: DType) -> Result<SpaceSpec, SpaceError> {
    let spec = SpaceSpec {
        shape: vec![],
        dtype,
        spec: Some(SpaceKind::Discrete(DiscreteSpec { n, start })),
    };
    crate::spaces::validate_space(&spec)?;
    Ok(spec)
}

pub(crate) fn validate_discrete_at(space: &SpaceSpec, path: &str) -> Result<(), SpaceError> {
    if !space.shape.is_empty() {
        return err_space!(path, "Discrete", "shape must be empty");
    }

    if space.dtype == DType::Unspecified {
        return err_space!(path, "Discrete", "dtype must be set");
    }
    match space.dtype {
        DType::Int64 | DType::Int32 | DType::Uint8 => {}
        other => {
            return err_space!(
                path,
                "Discrete",
                format!("Discrete.dtype must be an integer type; got {other:?}")
            );
        }
    }

    let d = match &space.spec {
        Some(SpaceKind::Discrete(d)) => d,
        _ => return err_space!(path, "Discrete", "spec.discrete must be set"),
    };

    if d.n <= 0 {
        return err_space!(path, "Discrete", "n must be > 0");
    }

    // Gymnasium allows any start (including negative). This is mostly a sanity check:
    // ensure start + (n-1) doesn't overflow i64 if someone later computes max value.
    let max = d
        .start
        .checked_add(d.n - 1)
        .ok_or_else(|| SpaceError::Invalid {
            path: path.to_string(),
            msg: "[Discrete] start + (n-1) overflowed i64".to_string(),
        })?;
    let _ = max;

    Ok(())
}

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
    //
    // Compare the offset in the *unsigned* domain. For any `val >= d.start`,
    // `val.wrapping_sub(d.start)` is the true non-negative offset reinterpreted
    // as bits, which `as u64` reads as the correct magnitude (e.g. start =
    // i64::MIN, val = 0 gives an offset of 2^63, which is positive — a signed
    // `< d.n` comparison would read it as i64::MIN and wrongly accept it).
    // `d.n` is validated `> 0`, so `d.n as u64` is its exact value.
    let in_range = val >= d.start && (val.wrapping_sub(d.start) as u64) < d.n as u64;
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

    #[test]
    fn test_discrete_range_starting_at_i64_min_rejects_far_values() {
        let space = DiscreteBuilder::new(4).start(i64::MIN).build().unwrap();

        // The four in-range values are accepted.
        assert!(contains(&space, &SpaceValue::Discrete(i64::MIN)).is_ok());
        assert!(contains(&space, &SpaceValue::Discrete(i64::MIN + 3)).is_ok());
        // Out-of-range values far above start must be rejected.
        assert!(contains(&space, &SpaceValue::Discrete(i64::MIN + 4)).is_err());
        assert!(contains(&space, &SpaceValue::Discrete(0)).is_err());
        assert!(contains(&space, &SpaceValue::Discrete(i64::MAX)).is_err());
    }
}
