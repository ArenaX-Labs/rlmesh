//! Human-readable, Gymnasium-style [`Display`] for space specs.
//!
//! Renders e.g. `Box(-1.0, 1.0, (3,), float32)`, `Discrete(4, start=-1)`, or a
//! nested `Dict('obs': Box(...), 'flag': Discrete(2))` — for error messages and
//! logs, where the `Debug` form is unreadable. Spec-shape only; this carries no
//! values.

use core::fmt;

use crate::dtype::DType;
use crate::scalar::decode_scalars;
use crate::types::{BoxBounds, BoxSpec, SpaceKind, SpaceSpec};

impl fmt::Display for SpaceSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.spec.as_ref() {
            None => write!(f, "Unspecified({})", fmt_shape(&self.shape)),
            Some(SpaceKind::Box(spec)) => {
                let (low, high) = box_bounds_summary(spec, self.dtype);
                write!(
                    f,
                    "Box({low}, {high}, {}, {})",
                    fmt_shape(&self.shape),
                    self.dtype
                )
            }
            Some(SpaceKind::Discrete(spec)) => {
                if spec.start == 0 {
                    write!(f, "Discrete({})", spec.n)
                } else {
                    write!(f, "Discrete({}, start={})", spec.n, spec.start)
                }
            }
            Some(SpaceKind::MultiBinary(_)) => {
                write!(f, "MultiBinary({})", fmt_shape(&self.shape))
            }
            Some(SpaceKind::MultiDiscrete(spec)) => write!(f, "MultiDiscrete({:?})", spec.nvec),
            Some(SpaceKind::Text(spec)) => {
                if spec.charset.is_empty() {
                    write!(f, "Text({}, {})", spec.min_length, spec.max_length)
                } else {
                    write!(
                        f,
                        "Text({}, {}, charset={:?})",
                        spec.min_length, spec.max_length, spec.charset
                    )
                }
            }
            Some(SpaceKind::Dict(spec)) => {
                write!(f, "Dict(")?;
                for (i, (key, child)) in spec.keys.iter().zip(spec.spaces.iter()).enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{key:?}: {child}")?;
                }
                write!(f, ")")
            }
            Some(SpaceKind::Tuple(spec)) => {
                write!(f, "Tuple(")?;
                for (i, child) in spec.spaces.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{child}")?;
                }
                write!(f, ")")
            }
        }
    }
}

/// Python-style shape tuple: `()`, `(3,)`, `(2, 3)`.
fn fmt_shape(shape: &[i64]) -> String {
    let mut out = String::from("(");
    for (i, dim) in shape.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        out.push_str(&dim.to_string());
    }
    if shape.len() == 1 {
        out.push(',');
    }
    out.push(')');
    out
}

/// A representative `(low, high)` string pair for a Box's bounds. Elementwise
/// and per-element typed bounds are abbreviated rather than expanded.
fn box_bounds_summary(spec: &BoxSpec, dtype: DType) -> (String, String) {
    match &spec.bounds {
        Some(BoxBounds::Uniform(bounds)) => (fmt_f64(bounds.low), fmt_f64(bounds.high)),
        Some(BoxBounds::TypedUniform(bounds)) => (
            typed_one(&bounds.low, dtype),
            typed_one(&bounds.high, dtype),
        ),
        Some(BoxBounds::Elementwise(_)) | Some(BoxBounds::TypedElementwise(_)) => {
            ("[…]".to_string(), "[…]".to_string())
        }
        Some(BoxBounds::Unbounded(_)) | None => ("-inf".to_string(), "inf".to_string()),
    }
}

fn typed_one(bytes: &[u8], dtype: DType) -> String {
    match decode_scalars(bytes, dtype) {
        Ok(scalars) if !scalars.is_empty() => fmt_f64(scalars[0].to_f64(dtype)),
        _ => "?".to_string(),
    }
}

/// Format a bound so integers read like floats (`-1.0`, not `-1`), matching the
/// Gymnasium repr, and infinities read as `inf`/`-inf`.
fn fmt_f64(value: f64) -> String {
    if value.is_infinite() {
        return if value < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    let rendered = format!("{value}");
    if rendered.contains('.') || rendered.contains('e') || rendered.contains("NaN") {
        rendered
    } else {
        format!("{rendered}.0")
    }
}

#[cfg(test)]
mod tests {
    use crate::DType;
    use crate::spaces::{
        BoxSpaceBuilder, DictSpaceBuilder, DiscreteBuilder, MultiBinaryBuilder,
        MultiDiscreteBuilder, TextBuilder, TupleSpaceBuilder,
    };

    #[test]
    fn renders_gym_style_specs() {
        let box_spec = BoxSpaceBuilder::scalar(-1.0, 1.0, vec![3])
            .dtype(DType::Float32)
            .build()
            .unwrap();
        assert_eq!(box_spec.to_string(), "Box(-1.0, 1.0, (3,), float32)");

        let unbounded = BoxSpaceBuilder::unbounded(vec![2, 2])
            .dtype(DType::Float64)
            .build()
            .unwrap();
        assert_eq!(unbounded.to_string(), "Box(-inf, inf, (2, 2), float64)");

        assert_eq!(
            DiscreteBuilder::new(4).build().unwrap().to_string(),
            "Discrete(4)"
        );
        assert_eq!(
            DiscreteBuilder::new(4)
                .start(-1)
                .build()
                .unwrap()
                .to_string(),
            "Discrete(4, start=-1)"
        );
        assert_eq!(
            MultiBinaryBuilder::shape(vec![8])
                .build()
                .unwrap()
                .to_string(),
            "MultiBinary((8,))"
        );
        assert_eq!(
            MultiDiscreteBuilder::vector(vec![3, 2])
                .build()
                .unwrap()
                .to_string(),
            "MultiDiscrete([3, 2])"
        );
    }

    #[test]
    fn renders_nested_composites() {
        let spec = DictSpaceBuilder::new()
            .insert("flag", DiscreteBuilder::new(2).build().unwrap())
            .insert(
                "obs",
                TupleSpaceBuilder::new()
                    .with(TextBuilder::new(8).build().unwrap())
                    .build()
                    .unwrap(),
            )
            .build()
            .unwrap();
        assert_eq!(
            spec.to_string(),
            "Dict(\"flag\": Discrete(2), \"obs\": Tuple(Text(1, 8)))"
        );
    }
}
