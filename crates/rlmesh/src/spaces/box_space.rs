//! `Box` and its bounds.

use std::fmt;

use rlmesh_spaces::scalar::{Scalar, decode_scalars};
use rlmesh_spaces::spaces::BoxSpaceBuilder;
use rlmesh_spaces::types::SpaceKind as WireKind;
use rlmesh_spaces::{BoxBounds, DType as CoreDType};

use super::{Space, SpaceSpec, space_newtype};
use crate::{DType, Error, Result};

/// One Box bound: a float, an integer, or infinity (`f64::INFINITY`).
/// Integers stay exact; `From` each primitive picks the variant.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Bound {
    /// A bool dtype's bound.
    Bool(bool),
    /// A signed integer.
    Int(i64),
    /// An unsigned integer (above `i64::MAX` for `uint64`).
    Uint(u64),
    /// A float, possibly infinite; never NaN in a space.
    Float(f64),
}

macro_rules! bound_from {
    ($variant:ident: $($ty:ty),*) => {$(
        impl From<$ty> for Bound {
            fn from(value: $ty) -> Self {
                Bound::$variant(value.into())
            }
        }
    )*};
}
bound_from!(Int: i8, i16, i32, i64);
bound_from!(Uint: u8, u16, u32, u64);
bound_from!(Float: f32, f64);

impl From<bool> for Bound {
    fn from(value: bool) -> Self {
        Bound::Bool(value)
    }
}

impl Bound {
    fn as_f64(self) -> f64 {
        match self {
            Bound::Bool(value) => f64::from(u8::from(value)),
            Bound::Int(value) => value as f64,
            Bound::Uint(value) => value as f64,
            Bound::Float(value) => value,
        }
    }

    /// The exact integer, when the bound is finite and integral.
    fn as_exact_int(self) -> Option<i128> {
        match self {
            Bound::Bool(value) => Some(i128::from(value)),
            Bound::Int(value) => Some(i128::from(value)),
            Bound::Uint(value) => Some(i128::from(value)),
            Bound::Float(value) if value.is_finite() && value.fract() == 0.0 => {
                // Inside +/-2^127, so the cast is exact for an integral f64.
                (value.abs() < 2f64.powi(127)).then_some(value as i128)
            }
            Bound::Float(_) => None,
        }
    }
}

impl fmt::Display for Bound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Bound::Bool(value) => write!(f, "{value}"),
            Bound::Int(value) => write!(f, "{value}"),
            Bound::Uint(value) => write!(f, "{value}"),
            Bound::Float(value) => write!(f, "{value:?}"),
        }
    }
}

/// A Box's `low` or `high`: one bound for every element, or one per element
/// (row-major).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum BoxBound {
    /// One bound for every element (`uniform` and `typed_uniform`).
    Uniform(Bound),
    /// One bound per element (`elementwise` and `typed_elementwise`).
    Elementwise(Vec<Bound>),
}

/// How a Box stores its bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BoundsKind {
    /// `"unbounded"`.
    Unbounded,
    /// `"uniform"`: one float pair.
    Uniform,
    /// `"elementwise"`: float pairs per element.
    Elementwise,
    /// `"typed_uniform"`: one pair in the dtype's domain.
    TypedUniform,
    /// `"typed_elementwise"`: pairs per element in the dtype's domain.
    TypedElementwise,
}

impl BoundsKind {
    /// The canonical spelling (`"typed_uniform"`).
    pub fn as_str(self) -> &'static str {
        match self {
            BoundsKind::Unbounded => "unbounded",
            BoundsKind::Uniform => "uniform",
            BoundsKind::Elementwise => "elementwise",
            BoundsKind::TypedUniform => "typed_uniform",
            BoundsKind::TypedElementwise => "typed_elementwise",
        }
    }
}

impl fmt::Display for BoundsKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// `Box`'s keyword arguments.
#[derive(Clone, Debug, PartialEq)]
pub struct BoxOptions {
    /// Element dtype (`"float32"`).
    pub dtype: DType,
}

impl Default for BoxOptions {
    fn default() -> Self {
        Self {
            dtype: DType::Float32,
        }
    }
}

/// A box of numbers in `[low, high]`.
#[derive(Clone)]
#[repr(transparent)]
pub struct Box(Space);

space_newtype!(Box, Box);

impl Box {
    /// `Box(low, high, shape, dtype)`.
    ///
    /// `(-inf, inf)` is unbounded for any dtype. On an integer or bool dtype
    /// finite bounds are stored exactly in the dtype's domain and must be
    /// integers (`0.5` is the ValueError class); otherwise they are floats.
    /// NaN is refused.
    pub fn new(
        low: impl Into<Bound>,
        high: impl Into<Bound>,
        shape: impl Into<Vec<i64>>,
        options: BoxOptions,
    ) -> Result<Box> {
        let spec = box_spec(low.into(), high.into(), shape.into(), options.dtype)?;
        Ok(Box(Space::new(spec)))
    }

    /// The lower bound: `None` for an unbounded Box.
    pub fn low(&self) -> Option<BoxBound> {
        self.bounds().map(|(low, _)| low)
    }

    /// The upper bound: `None` for an unbounded Box.
    pub fn high(&self) -> Option<BoxBound> {
        self.bounds().map(|(_, high)| high)
    }

    /// How the bounds are stored: `None` when the spec has no bounds.
    pub fn bounds_kind(&self) -> Option<BoundsKind> {
        Some(match self.wire_bounds()? {
            BoxBounds::Unbounded(_) => BoundsKind::Unbounded,
            BoxBounds::Uniform(_) => BoundsKind::Uniform,
            BoxBounds::Elementwise(_) => BoundsKind::Elementwise,
            BoxBounds::TypedUniform(_) => BoundsKind::TypedUniform,
            BoxBounds::TypedElementwise(_) => BoundsKind::TypedElementwise,
        })
    }

    fn wire_bounds(&self) -> Option<&BoxBounds> {
        match &self.spec().wire().spec {
            Some(WireKind::Box(spec)) => spec.bounds.as_ref(),
            _ => None,
        }
    }

    fn bounds(&self) -> Option<(BoxBound, BoxBound)> {
        let dtype = self.spec().wire().dtype;
        let typed = |bytes: &[u8]| -> Vec<Bound> {
            decode_scalars(bytes, dtype)
                .map(|scalars| scalars.into_iter().map(|s| typed_bound(s, dtype)).collect())
                .unwrap_or_default()
        };
        let floats = |values: &[f64]| values.iter().map(|v| Bound::Float(*v)).collect();
        Some(match self.wire_bounds()? {
            BoxBounds::Unbounded(_) => return None,
            BoxBounds::Uniform(b) => (
                BoxBound::Uniform(Bound::Float(b.low)),
                BoxBound::Uniform(Bound::Float(b.high)),
            ),
            BoxBounds::Elementwise(b) => (
                BoxBound::Elementwise(floats(&b.low)),
                BoxBound::Elementwise(floats(&b.high)),
            ),
            BoxBounds::TypedUniform(b) => {
                let (low, high) = (typed(&b.low), typed(&b.high));
                (
                    BoxBound::Uniform(*low.first()?),
                    BoxBound::Uniform(*high.first()?),
                )
            }
            BoxBounds::TypedElementwise(b) => (
                BoxBound::Elementwise(typed(&b.low)),
                BoxBound::Elementwise(typed(&b.high)),
            ),
        })
    }
}

/// A typed bound read in the dtype's domain: bool, unsigned or signed.
fn typed_bound(scalar: Scalar, dtype: CoreDType) -> Bound {
    match scalar {
        Scalar::Bool(value) => Bound::Bool(value),
        Scalar::Float(value) => Bound::Float(value),
        Scalar::Int(value) => match dtype {
            CoreDType::Bool => Bound::Bool(value != 0),
            CoreDType::Uint8 | CoreDType::Uint16 | CoreDType::Uint32 | CoreDType::Uint64 => {
                Bound::Uint(scalar.as_u64())
            }
            _ => Bound::Int(value),
        },
    }
}

/// Build a Box spec, validating shape, dtype and bounds.
pub(crate) fn box_spec(
    low: Bound,
    high: Bound,
    shape: Vec<i64>,
    dtype: DType,
) -> Result<SpaceSpec> {
    let (low_f, high_f) = (low.as_f64(), high.as_f64());
    if low_f.is_nan() || high_f.is_nan() {
        return Err(Error::invalid_argument(
            "Box bounds must be real numbers, not NaN",
        ));
    }
    let integral = dtype.is_integral();
    if integral {
        for (name, bound) in [("low", low), ("high", high)] {
            if let Bound::Float(value) = bound
                && value.is_finite()
                && value.fract() != 0.0
            {
                return Err(Error::invalid_argument(format!(
                    "Box {name} bound {value} is not an integer, but dtype {dtype} is integral"
                )));
            }
        }
    }
    let builder = if low_f == f64::NEG_INFINITY && high_f == f64::INFINITY {
        BoxSpaceBuilder::unbounded(shape)
    } else if integral && low_f.is_finite() && high_f.is_finite() {
        let (Some(low), Some(high)) = (low.as_exact_int(), high.as_exact_int()) else {
            return Err(Error::invalid_argument(format!(
                "Box bounds [{low}, {high}] do not fit a 64-bit integer range"
            )));
        };
        if let (Ok(low), Ok(high)) = (i64::try_from(low), i64::try_from(high)) {
            BoxSpaceBuilder::int_scalar(low, high, shape)
        } else if let (Ok(low), Ok(high)) = (u64::try_from(low), u64::try_from(high)) {
            BoxSpaceBuilder::uint_scalar(low, high, shape)
        } else {
            return Err(Error::invalid_argument(format!(
                "Box bounds [{low}, {high}] do not fit a 64-bit integer range"
            )));
        }
    } else {
        BoxSpaceBuilder::scalar(low_f, high_f, shape)
    };
    builder
        .dtype(dtype.to_core())
        .build()
        .map(SpaceSpec::from_wire)
        .map_err(|err| Error::invalid_argument(err.to_string()))
}

/// Build a per-element Box spec (the C/C++ `ext` builders): each pair
/// follows `box_spec`'s rules; all pairs must agree on float vs typed.
pub(crate) fn box_elementwise_spec(
    low: &[Bound],
    high: &[Bound],
    shape: Vec<i64>,
    dtype: DType,
) -> Result<SpaceSpec> {
    let numel = shape
        .iter()
        .try_fold(1usize, |acc, &dim| {
            usize::try_from(dim)
                .ok()
                .and_then(|dim| acc.checked_mul(dim))
        })
        .ok_or_else(|| Error::invalid_argument("Box shape must be non-negative"))?;
    if low.len() != numel || high.len() != numel {
        return Err(Error::invalid_argument(format!(
            "Box bounds need {numel} entries each (the shape's element count), got {} and {}",
            low.len(),
            high.len()
        )));
    }
    if low.iter().chain(high).any(|bound| bound.as_f64().is_nan()) {
        return Err(Error::invalid_argument(
            "Box bounds must be real numbers, not NaN",
        ));
    }
    let integral = dtype.is_integral();
    let all_finite = low.iter().chain(high).all(|b| b.as_f64().is_finite());
    let builder = if integral && all_finite {
        let to_ints = |bounds: &[Bound]| -> Result<Vec<i128>> {
            bounds
                .iter()
                .map(|bound| {
                    bound.as_exact_int().ok_or_else(|| {
                        Error::invalid_argument(format!(
                            "Box bound {bound} is not an integer, but dtype {dtype} is integral"
                        ))
                    })
                })
                .collect()
        };
        let (lows, highs) = (to_ints(low)?, to_ints(high)?);
        let signed = |v: &[i128]| {
            v.iter()
                .map(|&x| i64::try_from(x).ok())
                .collect::<Option<Vec<_>>>()
        };
        let unsigned = |v: &[i128]| {
            v.iter()
                .map(|&x| u64::try_from(x).ok())
                .collect::<Option<Vec<_>>>()
        };
        if let (Some(l), Some(h)) = (signed(&lows), signed(&highs)) {
            BoxSpaceBuilder::int_tensor(l, h, shape)
        } else if let (Some(l), Some(h)) = (unsigned(&lows), unsigned(&highs)) {
            BoxSpaceBuilder::uint_tensor(l, h, shape)
        } else {
            return Err(Error::invalid_argument(
                "Box bounds do not fit a 64-bit integer range",
            ));
        }
    } else {
        if integral
            && let Some(bound) = low.iter().chain(high).find(|b| {
                let v = b.as_f64();
                v.is_finite() && v.fract() != 0.0
            })
        {
            return Err(Error::invalid_argument(format!(
                "Box bound {bound} is not an integer, but dtype {dtype} is integral"
            )));
        }
        BoxSpaceBuilder::tensor(
            low.iter().map(|b| b.as_f64()).collect(),
            high.iter().map(|b| b.as_f64()).collect(),
            shape,
        )
    };
    builder
        .dtype(dtype.to_core())
        .build()
        .map(SpaceSpec::from_wire)
        .map_err(|err| Error::invalid_argument(err.to_string()))
}
