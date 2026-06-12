use crate::errors::{SpaceError, err_space};
use crate::spaces::{SpaceSpec, space_spec, validate_space};
use crate::{DType, MatrixInt, MultiDiscreteSpec, VectorInt, multi_discrete_spec};

pub struct MultiDiscreteBuilder {
    dtype: DType,
    shape: Vec<i64>,
    nvec: multi_discrete_spec::Nvec,
}

impl MultiDiscreteBuilder {
    /// `MultiDiscrete(nvec: [n0, n1, ...])` sets shape to `[len]`.
    pub fn vector(nvec: impl Into<Vec<i64>>) -> Self {
        let nvec = nvec.into();

        Self {
            shape: vec![nvec.len() as i64],
            dtype: DType::Int64,
            nvec: multi_discrete_spec::Nvec::Flat(VectorInt { data: nvec }),
        }
    }

    /// `MultiDiscrete(nvec: [[...], [...]])` sets shape to `[rows, cols]`.
    pub fn matrix(rows: impl Into<Vec<Vec<i64>>>) -> Self {
        let rows = rows.into();
        let r = rows.len();
        let c = rows.first().map(|x| x.len()).unwrap_or(0);

        Self {
            shape: vec![r as i64, c as i64],
            dtype: DType::Int64,
            nvec: multi_discrete_spec::Nvec::Shaped(MatrixInt {
                data: rows
                    .into_iter()
                    .map(|row| VectorInt { data: row })
                    .collect(),
            }),
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
            spec: Some(space_spec::Spec::MultiDiscrete(MultiDiscreteSpec {
                nvec: Some(self.nvec),
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

    for (i, &d) in space.shape.iter().enumerate() {
        if d <= 0 {
            return err_space!(
                path,
                "MultiDiscrete",
                format!("MultiDiscrete.shape[{i}] must be > 0")
            );
        }
    }

    let md = match &space.spec {
        Some(space_spec::Spec::MultiDiscrete(md)) => md,
        _ => {
            return err_space!(path, "MultiDiscrete", "spec.multi_discrete must be set");
        }
    };

    let nvec = match &md.nvec {
        Some(nvec) => nvec,
        None => return err_space!(path, "MultiDiscrete", "nvec must be set"),
    };

    match nvec {
        // rank-1 nvec
        multi_discrete_spec::Nvec::Flat(v) => {
            let values = &v.data;

            if values.is_empty() {
                return err_space!(path, "MultiDiscrete", "nvec.flat.data must be non-empty");
            }
            for (i, &n) in values.iter().enumerate() {
                if n <= 0 {
                    return err_space!(
                        path,
                        "MultiDiscrete",
                        format!("nvec.flat.data[{i}] must be > 0")
                    );
                }
            }

            // canonical shape for flat form: [len(values)]
            if space.shape.len() != 1 || space.shape[0] != values.len() as i64 {
                return err_space!(
                    path,
                    "MultiDiscrete",
                    "shape mismatch: for flat nvec, expected shape == [len(nvec)]"
                );
            }
            Ok(())
        }

        // rank-2 nvec (matrix)
        multi_discrete_spec::Nvec::Shaped(mv) => {
            let rows = &mv.data;
            if rows.is_empty() {
                return err_space!(path, "MultiDiscrete", "nvec.shaped.data must be non-empty");
            }

            let cols = rows[0].data.len();
            if cols == 0 {
                return err_space!(path, "MultiDiscrete", "nvec.shaped rows must be non-empty");
            }

            // must be rectangular
            for (ri, r) in rows.iter().enumerate() {
                if r.data.len() != cols {
                    return err_space!(
                        path,
                        "MultiDiscrete",
                        format!("nvec.shaped row {ri} length mismatch")
                    );
                }
            }

            // all entries > 0
            for (ri, r) in rows.iter().enumerate() {
                for (ci, &n) in r.data.iter().enumerate() {
                    if n <= 0 {
                        return err_space!(
                            path,
                            "MultiDiscrete",
                            format!("nvec.shaped[{ri}][{ci}] must be > 0")
                        );
                    }
                }
            }

            // canonical shape for matrix form: [rows, cols]
            if space.shape.len() != 2
                || space.shape[0] != rows.len() as i64
                || space.shape[1] != cols as i64
            {
                return err_space!(
                    path,
                    "MultiDiscrete",
                    "MultiDiscrete shape mismatch: expected shape == [rows, cols] for shaped"
                );
            }

            Ok(())
        }
    }
}
