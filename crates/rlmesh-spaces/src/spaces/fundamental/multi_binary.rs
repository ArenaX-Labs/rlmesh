use crate::errors::{SpaceError, err_space};
use crate::spaces::{SpaceKind, SpaceSpec, SpaceValue, validate_space};
use crate::{DType, MultiBinarySpec};

#[must_use = "a space builder does nothing until .build() is called"]
pub struct MultiBinaryBuilder {
    shape: Vec<i64>,
    dtype: DType,
}

impl MultiBinaryBuilder {
    /// `MultiBinary(n: int)` sets shape to `[n]`.
    pub fn scalar(n: i64) -> Self {
        Self {
            shape: vec![n],
            dtype: DType::Uint8,
        }
    }

    /// `MultiBinary(shape: [d0, d1, ...])`.
    pub fn shape(shape: impl Into<Vec<i64>>) -> Self {
        Self {
            shape: shape.into(),
            dtype: DType::Uint8,
        }
    }

    pub fn dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }

    pub fn build(self) -> Result<SpaceSpec, SpaceError> {
        let spec = SpaceSpec {
            shape: self.shape,
            dtype: self.dtype,
            spec: Some(SpaceKind::MultiBinary(MultiBinarySpec)),
        };

        validate_space(&spec)?;
        Ok(spec)
    }
}

pub(crate) fn validate_multibinary_at(spec: &SpaceSpec, path: &str) -> Result<(), SpaceError> {
    if spec.shape.is_empty() {
        return err_space!(path, "MultiBinary", "shape must be set (rank >= 1)");
    }
    if spec.dtype == DType::Unspecified {
        return err_space!(path, "MultiBinary", "dtype must be set");
    }
    // MultiBinary crosses the wire as one byte per bit, so its dtype must be a
    // single-byte width (the canonical `uint8`, or `bool`/`int8`). A wider dtype
    // would make the leaf byte count (`numel`) disagree with the raw batch
    // stride (`numel * dtype_size`) and with the proto's "raw element bytes for
    // the child dtype" contract.
    if crate::dtype_size(spec.dtype) != 1 {
        return err_space!(
            path,
            "MultiBinary",
            format!(
                "dtype must be a single-byte type (bool/uint8/int8), got {}",
                spec.dtype
            )
        );
    }

    for (i, &d) in spec.shape.iter().enumerate() {
        if d <= 0 {
            return err_space!(path, "MultiBinary", format!("shape[{i}] must be > 0"));
        }
    }

    // The MultiBinary marker carries no fields; the dimensions live entirely in
    // `SpaceSpec.shape`, validated above.
    if !matches!(&spec.spec, Some(SpaceKind::MultiBinary(_))) {
        return err_space!(path, "MultiBinary", "spec.multi_binary must be set");
    }

    Ok(())
}

pub(crate) fn contains_multibinary(
    space: &SpaceSpec,
    value: &SpaceValue,
    path: &str,
) -> Result<(), SpaceError> {
    let vals = match value {
        SpaceValue::MultiBinary(v) => v,
        _ => return err_space!(path, "expected MultiBinary value"),
    };

    if !matches!(&space.spec, Some(SpaceKind::MultiBinary(_))) {
        return err_space!(path, "space is not MultiBinary");
    }

    // The number of bits is the product of the shape dimensions. Fold with
    // checked conversion/multiplication so a negative dim (which would cast to a
    // huge `usize`) or an overflowing product is reported, never panicked on.
    let expected_size: usize = space
        .shape
        .iter()
        .try_fold(1usize, |acc, &d| {
            usize::try_from(d).ok().and_then(|d| acc.checked_mul(d))
        })
        .ok_or_else(|| {
            SpaceError::invalid(
                path,
                "MultiBinary shape has an invalid or too-large dimension",
            )
        })?;

    if vals.len() != expected_size {
        return err_space!(
            path,
            format!(
                "MultiBinary size mismatch: expected {}, got {}",
                expected_size,
                vals.len()
            )
        );
    }

    // Values are bools, always valid
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::spaces::fundamental::MultiBinaryBuilder;
    use crate::spaces::{SpaceValue, contains};

    #[test]
    fn test_multibinary_contains() {
        let space = MultiBinaryBuilder::scalar(3).build().unwrap();

        assert!(contains(&space, &SpaceValue::MultiBinary(vec![true, false, true])).is_ok());
        assert!(contains(&space, &SpaceValue::MultiBinary(vec![true, false])).is_err());
        assert!(contains(&space, &SpaceValue::Discrete(1)).is_err());
    }

    #[test]
    fn test_multibinary_requires_single_byte_dtype() {
        use crate::DType;

        // MultiBinary crosses the wire as one byte per bit, so a wider dtype is
        // rejected (it would desync the leaf byte count from the batch stride).
        assert!(
            MultiBinaryBuilder::scalar(3)
                .dtype(DType::Int32)
                .build()
                .is_err()
        );
        assert!(
            MultiBinaryBuilder::scalar(3)
                .dtype(DType::Uint16)
                .build()
                .is_err()
        );
        // The 1-byte dtypes are accepted.
        for dtype in [DType::Uint8, DType::Int8, DType::Bool] {
            assert!(
                MultiBinaryBuilder::scalar(3).dtype(dtype).build().is_ok(),
                "{dtype} should be a valid MultiBinary dtype"
            );
        }
    }
}
