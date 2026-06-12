use crate::DType;
use crate::errors::{SpaceError, err_space};
use crate::spaces::{SpaceSpec, space_spec};
use crate::{AxiswiseBounds, BoxSpec, ElementwiseBounds, UniformBounds, box_spec};

pub struct BoxSpaceBuilder {
    shape: Vec<i64>,
    dtype: DType,
    bounds: box_spec::Bounds,
}

impl BoxSpaceBuilder {
    pub fn unbounded(shape: impl Into<Vec<i64>>) -> Self {
        Self {
            shape: shape.into(),
            dtype: DType::Float32,
            bounds: box_spec::Bounds::Unbounded(true),
        }
    }

    pub fn scalar(low: f64, high: f64, shape: impl Into<Vec<i64>>) -> Self {
        Self {
            shape: shape.into(),
            dtype: DType::Float32,
            bounds: box_spec::Bounds::Uniform(UniformBounds { low, high }),
        }
    }

    pub fn per_axis(low: Vec<f64>, high: Vec<f64>, shape: impl Into<Vec<i64>>) -> Self {
        Self {
            shape: shape.into(),
            dtype: DType::Float32,
            bounds: box_spec::Bounds::Axiswise(AxiswiseBounds { low, high }),
        }
    }

    pub fn tensor(low: Vec<f64>, high: Vec<f64>, shape: impl Into<Vec<i64>>) -> Self {
        Self {
            shape: shape.into(),
            dtype: DType::Float32,
            bounds: box_spec::Bounds::Elementwise(ElementwiseBounds { low, high }),
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
            spec: Some(space_spec::Spec::Box(BoxSpec {
                bounds: Some(self.bounds),
            })),
        };
        crate::spaces::validate_space(&spec)?;
        Ok(spec)
    }
}

pub(crate) fn validate_box_at(space: &SpaceSpec, path: &str) -> Result<(), SpaceError> {
    if space.shape.is_empty() {
        return err_space!(path, "Box", "shape must be set (rank >= 1)");
    }

    if space.dtype == DType::Unspecified {
        return err_space!(path, "Box", "dtype must be set");
    }

    for (i, &d) in space.shape.iter().enumerate() {
        if d <= 0 {
            return err_space!(path, "Box", format!("shape[{i}] must be > 0"));
        }
    }

    let b = match &space.spec {
        Some(space_spec::Spec::Box(b)) => b,
        _ => return err_space!(path, "Box", "spec.box must be set"),
    };

    let rank = space.shape.len();
    let numel: usize = space
        .shape
        .iter()
        .try_fold(1usize, |acc, &d| (d as usize).checked_mul(acc))
        .ok_or_else(|| SpaceError::Invalid {
            path: path.to_string(),
            msg: "Box.shape product overflowed".to_string(),
        })?;

    match &b.bounds {
        Some(box_spec::Bounds::Unbounded(_)) => Ok(()),

        Some(box_spec::Bounds::Uniform(s)) => {
            if s.low > s.high {
                return err_space!(path, "Box", "scalar bounds invalid: low > high");
            }
            Ok(())
        }

        // per-axis / broadcast: len == rank
        Some(box_spec::Bounds::Axiswise(v)) => {
            if v.low.len() != v.high.len() {
                return err_space!(
                    path,
                    "Box",
                    "per-axis bounds invalid: low/high length mismatch"
                );
            }
            if v.low.len() != rank {
                return err_space!(
                    path,
                    "Box",
                    format!("per-axis bounds invalid: expected length {rank}")
                );
            }
            for i in 0..rank {
                if v.low[i] > v.high[i] {
                    return err_space!(
                        path,
                        "Box",
                        format!("per-axis bounds invalid: low>high at axis {i}")
                    );
                }
            }
            Ok(())
        }

        // elementwise / tensor: len == numel
        Some(box_spec::Bounds::Elementwise(t)) => {
            if t.low.len() != t.high.len() {
                return err_space!(
                    path,
                    "Box",
                    "tensor bounds invalid: low/high length mismatch"
                );
            }
            if t.low.len() != numel {
                return err_space!(
                    path,
                    "Box",
                    format!("tensor bounds invalid: expected length {numel}")
                );
            }
            for i in 0..numel {
                if t.low[i] > t.high[i] {
                    return err_space!(
                        path,
                        "Box",
                        format!("tensor bounds invalid: low>high at element {i}")
                    );
                }
            }
            Ok(())
        }

        None => err_space!(path, "Box", "bounds must be set"),
    }
}
