use crate::errors::{SpaceError, err_space};
use crate::spaces::{SpaceKind, SpaceSpec};
use crate::{DType, DiscreteSpec};

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
