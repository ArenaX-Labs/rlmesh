use crate::errors::{SpaceError, err_space};
use crate::scalar::check_int_in_dtype_range;
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

    // The whole range [start, start + n - 1] must fit the wire dtype.
    let max = d
        .start
        .checked_add(d.n - 1)
        .ok_or_else(|| SpaceError::Invalid {
            path: path.to_string(),
            msg: "[Discrete] start + (n-1) overflowed i64".to_string(),
        })?;
    for bound in [d.start, max] {
        if check_int_in_dtype_range(bound, space.dtype).is_err() {
            return err_space!(
                path,
                "Discrete",
                format!("value {bound} does not fit dtype {}", space.dtype)
            );
        }
    }

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

    // Offset in the unsigned domain so a range ending at i64::MAX does not overflow.
    let in_range = val >= d.start && (val.wrapping_sub(d.start) as u64) < d.n as u64;
    if !in_range {
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
    use crate::DType;
    use crate::spaces::fundamental::DiscreteBuilder;
    use crate::spaces::{SpaceValue, contains};

    #[test]
    fn range_must_fit_declared_dtype() {
        assert!(
            DiscreteBuilder::new(10)
                .start(250)
                .dtype(DType::Uint8)
                .build()
                .is_err()
        );
        assert!(
            DiscreteBuilder::new(256)
                .dtype(DType::Uint8)
                .build()
                .is_ok()
        );
        assert!(
            DiscreteBuilder::new(2)
                .start(-1)
                .dtype(DType::Uint8)
                .build()
                .is_err()
        );
    }

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
        let space = DiscreteBuilder::new(4).start(i64::MAX - 3).build().unwrap();

        assert!(contains(&space, &SpaceValue::Discrete(i64::MAX - 3)).is_ok());
        assert!(contains(&space, &SpaceValue::Discrete(i64::MAX)).is_ok());
        assert!(contains(&space, &SpaceValue::Discrete(i64::MAX - 4)).is_err());
    }

    #[test]
    fn test_discrete_range_starting_at_i64_min_rejects_far_values() {
        let space = DiscreteBuilder::new(4).start(i64::MIN).build().unwrap();

        assert!(contains(&space, &SpaceValue::Discrete(i64::MIN)).is_ok());
        assert!(contains(&space, &SpaceValue::Discrete(i64::MIN + 3)).is_ok());
        assert!(contains(&space, &SpaceValue::Discrete(i64::MIN + 4)).is_err());
        assert!(contains(&space, &SpaceValue::Discrete(0)).is_err());
        assert!(contains(&space, &SpaceValue::Discrete(i64::MAX)).is_err());
    }
}
