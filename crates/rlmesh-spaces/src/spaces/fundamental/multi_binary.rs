use crate::errors::{SpaceError, err_space};
use crate::spaces::{SpaceKind, SpaceSpec, SpaceValue, validate_space};
use crate::{DType, MultiBinaryDims, MultiBinarySpec};

#[must_use = "a space builder does nothing until .build() is called"]
pub struct MultiBinaryBuilder {
    shape: Vec<i64>,
    dtype: DType,
    n: MultiBinaryDims,
}

impl MultiBinaryBuilder {
    /// `MultiBinary(n: int)` sets shape to `[n]`.
    pub fn scalar(n: i64) -> Self {
        Self {
            shape: vec![n],
            dtype: DType::Uint8,
            n: MultiBinaryDims::Size(n),
        }
    }

    /// `MultiBinary(shape: [d0, d1, ...])`.
    pub fn shape(shape: impl Into<Vec<i64>>) -> Self {
        let shape = shape.into();
        Self {
            shape: shape.clone(),
            dtype: DType::Uint8,
            n: MultiBinaryDims::Dims(shape),
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
            spec: Some(SpaceKind::MultiBinary(MultiBinarySpec { n: Some(self.n) })),
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

    for (i, &d) in spec.shape.iter().enumerate() {
        if d <= 0 {
            return err_space!(path, "MultiBinary", format!("shape[{i}] must be > 0"));
        }
    }

    let mb = match &spec.spec {
        Some(SpaceKind::MultiBinary(mb)) => mb,
        _ => {
            return err_space!(path, "MultiBinary", "spec.multi_binary must be set");
        }
    };

    let n = match &mb.n {
        Some(n) => n,
        None => return err_space!(path, "MultiBinary", "n must be set"),
    };

    match n {
        MultiBinaryDims::Size(n) => {
            if n <= &0 {
                return err_space!(path, "MultiBinary", "n.size must be > 0");
            }
            if spec.shape.len() != 1 || spec.shape[0] != *n {
                return err_space!(
                    path,
                    "MultiBinary",
                    "shape mismatch: for size n, expected shape == [n]"
                );
            }
            Ok(())
        }

        MultiBinaryDims::Dims(v) => {
            let dims = v;

            if dims.is_empty() {
                return err_space!(path, "MultiBinary", "n.dims must be non-empty");
            }
            for (i, &d) in dims.iter().enumerate() {
                if d <= 0 {
                    return err_space!(
                        path,
                        "MultiBinary",
                        format!("MultiBinarySpec.n.dims.data[{i}] must be > 0")
                    );
                }
            }
            if *dims != spec.shape {
                return err_space!(
                    path,
                    "MultiBinary",
                    "shape mismatch: expected SpaceSpec.shape == MultiBinarySpec.n.vector.data"
                );
            }
            Ok(())
        }
    }
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

    let mb = match &space.spec {
        Some(SpaceKind::MultiBinary(mb)) => mb,
        _ => return err_space!(path, "space is not MultiBinary"),
    };

    // Get expected size from the space
    let expected_size = match &mb.n {
        Some(MultiBinaryDims::Size(n)) => *n as usize,
        Some(MultiBinaryDims::Dims(dims)) => dims.iter().map(|&d| d as usize).product(),
        None => return err_space!(path, "MultiBinary.n not set"),
    };

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
}
