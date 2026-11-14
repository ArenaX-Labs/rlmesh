use crate::errors::{SpaceError, err_space};
use crate::scalar::check_int_in_dtype_range;
use crate::spaces::{SpaceKind, SpaceSpec, SpaceValue, validate_space};
use crate::{DType, MultiDiscreteSpec};

#[must_use = "a space builder does nothing until .build() is called"]
pub struct MultiDiscreteBuilder {
    dtype: DType,
    shape: Vec<i64>,
    nvec: Vec<i64>,
    start: Vec<i64>,
    /// Deferred construction error (e.g. a ragged `matrix(..)`), surfaced by
    /// [`MultiDiscreteBuilder::build`]. `matrix` cannot return a `Result`, so a
    /// non-rectangular input is recorded here and fails the build rather than
    /// being silently flattened into a different rectangle.
    error: Option<SpaceError>,
}

impl MultiDiscreteBuilder {
    /// `MultiDiscrete(nvec: [n0, n1, ...])` sets shape to `[len]`.
    pub fn vector(nvec: impl Into<Vec<i64>>) -> Self {
        let nvec = nvec.into();

        Self {
            shape: vec![nvec.len() as i64],
            dtype: DType::Int64,
            nvec,
            start: Vec::new(),
            error: None,
        }
    }

    /// `MultiDiscrete(nvec: [[...], [...]])` flattens the rows row-major and sets
    /// shape to `[rows, cols]`.
    pub fn matrix(rows: impl Into<Vec<Vec<i64>>>) -> Self {
        let rows = rows.into();
        let r = rows.len();
        let c = rows.first().map(|x| x.len()).unwrap_or(0);

        // A ragged matrix would flatten into an `nvec` whose length can still
        // equal `r * c` (e.g. row lengths [2, 1, 3] -> shape [3, 2], 6 entries),
        // silently reinterpreting the category counts as a different rectangle.
        // Record a non-rectangular input as a deferred error so `build` fails it.
        let error = rows.iter().position(|row| row.len() != c).map(|i| {
            SpaceError::invalid(
                "MultiDiscrete",
                format!(
                    "matrix rows must be rectangular: row {i} has {} columns, expected {c}",
                    rows[i].len()
                ),
            )
        });

        Self {
            shape: vec![r as i64, c as i64],
            dtype: DType::Int64,
            nvec: rows.into_iter().flatten().collect(),
            start: Vec::new(),
            error,
        }
    }

    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    /// Each element's lowest value, flat (row-major) as `nvec`.
    pub fn start(mut self, start: impl Into<Vec<i64>>) -> Self {
        self.start = start.into();
        self
    }

    pub fn build(self) -> Result<SpaceSpec, SpaceError> {
        if let Some(error) = self.error {
            return Err(error);
        }
        let spec = SpaceSpec {
            shape: self.shape,
            dtype: self.dtype,
            spec: Some(SpaceKind::MultiDiscrete(MultiDiscreteSpec {
                nvec: self.nvec,
                start: self.start,
            })),
        };

        validate_space(&spec)?;
        Ok(spec)
    }
}

pub(crate) fn validate_multidiscrete_at(space: &SpaceSpec, path: &str) -> Result<(), SpaceError> {
    if space.shape.is_empty() {
        return err_space!(path, "MultiDiscrete", "shape must be set (rank >= 1)");
    }
    if space.dtype == DType::Unspecified {
        return err_space!(path, "MultiDiscrete", "dtype must be set");
    }
    // Category indices are integers carried in the space dtype; a float dtype
    // would route them through `f16`/`f32` storage and silently lose precision
    // on the wire, so only integer dtypes are valid.
    if !space.dtype.is_integer() {
        return err_space!(
            path,
            "MultiDiscrete",
            format!("dtype must be an integer type, got {}", space.dtype)
        );
    }

    let mut numel: i64 = 1;
    for (i, &d) in space.shape.iter().enumerate() {
        if d <= 0 {
            return err_space!(
                path,
                "MultiDiscrete",
                format!("MultiDiscrete.shape[{i}] must be > 0")
            );
        }
        numel = numel
            .checked_mul(d)
            .ok_or_else(|| SpaceError::invalid(path, "MultiDiscrete.shape product overflowed"))?;
    }

    let md = match &space.spec {
        Some(SpaceKind::MultiDiscrete(md)) => md,
        _ => {
            return err_space!(path, "MultiDiscrete", "spec.multi_discrete must be set");
        }
    };

    if md.nvec.is_empty() {
        return err_space!(path, "MultiDiscrete", "nvec must be non-empty");
    }
    if !md.start.is_empty() && md.start.len() != md.nvec.len() {
        return err_space!(
            path,
            "MultiDiscrete",
            "shape mismatch: expected len(start) == len(nvec)"
        );
    }
    for (i, &n) in md.nvec.iter().enumerate() {
        if n <= 0 {
            return err_space!(path, "MultiDiscrete", format!("nvec[{i}] must be > 0"));
        }
        // Element i's values span start[i]..=start[i]+nvec[i]-1; both ends must
        // fit the declared dtype, or a valid value silently wraps on the wire
        // (e.g. MultiDiscrete(uint8, nvec=[1000]) would wrap 999). Reject at
        // construction.
        let low = md.start(i);
        let high = low.checked_add(n - 1);
        let fits = |v: i64| check_int_in_dtype_range(v, space.dtype).is_ok();
        if !fits(low) || !high.is_some_and(fits) {
            return err_space!(
                path,
                "MultiDiscrete",
                format!(
                    "element {i}'s values {low}..{low}+{n} do not fit dtype {}",
                    space.dtype
                )
            );
        }
    }

    // `nvec` is the flat (row-major) category counts; it must hold exactly one
    // entry per element of the logical shape.
    if md.nvec.len() as i64 != numel {
        return err_space!(
            path,
            "MultiDiscrete",
            "shape mismatch: expected len(nvec) == numel(shape)"
        );
    }

    Ok(())
}

pub(crate) fn contains_multidiscrete(
    space: &SpaceSpec,
    value: &SpaceValue,
    path: &str,
) -> Result<(), SpaceError> {
    let vals = match value {
        SpaceValue::MultiDiscrete(v) => v,
        _ => return err_space!(path, "expected MultiDiscrete value"),
    };

    let md = match &space.spec {
        Some(SpaceKind::MultiDiscrete(md)) => md,
        _ => return err_space!(path, "space is not MultiDiscrete"),
    };

    // `nvec` is the flat (row-major) per-element category counts.
    let nvec = &md.nvec;

    if vals.len() != nvec.len() {
        return err_space!(
            path,
            format!(
                "MultiDiscrete size mismatch: expected {}, got {}",
                nvec.len(),
                vals.len()
            )
        );
    }

    // Check each value is in range [start[i], start[i] + nvec[i])
    for (i, (&val, &n)) in vals.iter().zip(nvec.iter()).enumerate() {
        let low = md.start(i);
        if val < low || val - low >= n {
            return err_space!(
                path,
                format!("value[{}] = {} not in range [{low}, {low} + {})", i, val, n)
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::DType;
    use crate::spaces::fundamental::MultiDiscreteBuilder;
    use crate::spaces::{SpaceValue, contains};

    #[test]
    fn nvec_max_index_must_fit_declared_dtype() {
        // max index 999 does not fit u8 -> reject at construction, not a silent wrap.
        assert!(
            MultiDiscreteBuilder::vector(vec![1000])
                .dtype(DType::Uint8)
                .build()
                .is_err()
        );
        // max index 255 fits u8 exactly -> ok.
        assert!(
            MultiDiscreteBuilder::vector(vec![256])
                .dtype(DType::Uint8)
                .build()
                .is_ok()
        );
    }

    #[test]
    fn test_multidiscrete_contains() {
        let space = MultiDiscreteBuilder::vector(vec![2, 3]).build().unwrap();

        assert!(contains(&space, &SpaceValue::MultiDiscrete(vec![0, 2])).is_ok());
        assert!(contains(&space, &SpaceValue::MultiDiscrete(vec![1])).is_err());
        assert!(contains(&space, &SpaceValue::MultiDiscrete(vec![2, 0])).is_err());
        assert!(contains(&space, &SpaceValue::Discrete(1)).is_err());
    }

    #[test]
    fn a_start_shifts_the_values_and_must_fit_the_dtype() {
        let space = MultiDiscreteBuilder::vector(vec![3, 4])
            .start(vec![-1, 5])
            .build()
            .unwrap();
        assert!(contains(&space, &SpaceValue::MultiDiscrete(vec![-1, 8])).is_ok());
        assert!(contains(&space, &SpaceValue::MultiDiscrete(vec![2, 5])).is_err());
        assert!(contains(&space, &SpaceValue::MultiDiscrete(vec![0, 4])).is_err());
        let sample = crate::sample_seeded(&space, 3).unwrap();
        assert!(contains(&space, &sample).is_ok());
        for bad in [vec![-1], vec![250, 0]] {
            assert!(
                MultiDiscreteBuilder::vector(vec![10, 10])
                    .dtype(DType::Uint8)
                    .start(bad)
                    .build()
                    .is_err()
            );
        }
    }

    #[test]
    fn test_multidiscrete_requires_integer_dtype() {
        use crate::DType;

        // Indices are integers carried in the dtype; a float dtype would route
        // them through float storage and lose precision on the wire.
        assert!(
            MultiDiscreteBuilder::vector(vec![2, 3])
                .dtype(DType::Float32)
                .build()
                .is_err()
        );
        assert!(
            MultiDiscreteBuilder::vector(vec![2, 3])
                .dtype(DType::Int32)
                .build()
                .is_ok()
        );
    }

    #[test]
    fn test_multidiscrete_matrix_rejects_ragged_rows() {
        // Row lengths [2, 1, 3] sum to 6 == 3 * 2, so a naive flatten would pass
        // validation as a [3, 2] matrix while reinterpreting the category counts.
        // Raggedness must be rejected at build, not silently re-rectangled.
        let err = MultiDiscreteBuilder::matrix(vec![vec![2, 2], vec![3], vec![4, 4, 4]]).build();
        assert!(err.is_err(), "ragged matrix rows must be rejected");

        // A genuinely rectangular matrix still builds.
        assert!(
            MultiDiscreteBuilder::matrix(vec![vec![2, 3], vec![4, 5]])
                .build()
                .is_ok()
        );
    }

    #[test]
    fn test_multidiscrete_shape_product_overflow_is_rejected_not_panicked() {
        use crate::spaces::{SpaceKind, SpaceSpec};
        use crate::{DType, MultiDiscreteSpec, spaces::validate_space};

        // A shape whose product overflows i64 must be reported, never panic.
        let spec = SpaceSpec {
            shape: vec![i64::MAX, 2],
            dtype: DType::Int64,
            spec: Some(SpaceKind::MultiDiscrete(MultiDiscreteSpec {
                nvec: vec![1, 1],
                start: Vec::new(),
            })),
        };
        assert!(validate_space(&spec).is_err());
    }
}
