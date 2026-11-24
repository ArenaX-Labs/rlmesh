use std::cmp::Ordering;

use half::f16;
use thiserror::Error;

use crate::dtype::{DType, dtype_size};

/// A dtype-independent scalar element decoded from or encoded into tensor
/// bytes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    Bool(bool),
    Int(i64),
    Float(f64),
}

impl Scalar {
    /// Lossy integer view: `Bool` maps to 0/1, `Float` truncates toward zero
    /// (saturating at the `i64` range).
    pub fn as_i64(self) -> i64 {
        match self {
            Scalar::Bool(value) => i64::from(value),
            Scalar::Int(value) => value,
            Scalar::Float(value) => value as i64,
        }
    }

    /// Reinterpret an integer scalar's bit pattern as `u64` for the one dtype
    /// (`Uint64`) whose values do not fit `i64`, undoing the bit-cast
    /// [`decode_scalars`] stores them with; `Bool` maps to 0/1 and `Float`
    /// saturates.
    pub fn as_u64(self) -> u64 {
        match self {
            Scalar::Bool(value) => u64::from(value),
            Scalar::Int(value) => value as u64,
            Scalar::Float(value) => value as u64,
        }
    }

    /// Dtype-aware `f64` view: for `Uint64` the stored `i64` is reinterpreted
    /// as `u64` first, so values above `i64::MAX` convert to the correct
    /// positive magnitude.
    pub fn to_f64(self, dtype: DType) -> f64 {
        match self {
            Scalar::Bool(value) => f64::from(u8::from(value)),
            Scalar::Float(value) => value,
            Scalar::Int(value) if dtype == DType::Uint64 => (value as u64) as f64,
            Scalar::Int(value) => value as f64,
        }
    }

    /// Total order between two decoded scalars in their dtype's native domain.
    ///
    /// Integer dtypes compare as integers (`Uint64` unsigned, the rest signed)
    /// and float dtypes as floats. A `Float` bound against an `Int` value is
    /// compared exactly by real-number value, never by truncating either side.
    /// Returns `None` only when a `Float` operand is NaN.
    pub fn cmp_typed(self, other: Scalar, dtype: DType) -> Option<Ordering> {
        match (self, other) {
            (Scalar::Float(a), Scalar::Float(b)) => a.partial_cmp(&b),
            (Scalar::Bool(a), Scalar::Bool(b)) => Some(a.cmp(&b)),
            (Scalar::Float(f), other) => cmp_float_int(f, other.int_operand(dtype)),
            (this, Scalar::Float(f)) => {
                cmp_float_int(f, this.int_operand(dtype)).map(Ordering::reverse)
            }
            (a, b) if dtype == DType::Uint64 => Some(a.as_u64().cmp(&b.as_u64())),
            (a, b) => Some(a.as_i64().cmp(&b.as_i64())),
        }
    }

    fn int_operand(self, dtype: DType) -> IntOperand {
        match self {
            Scalar::Float(_) => unreachable!("int_operand called on a float scalar"),
            _ if dtype == DType::Uint64 => IntOperand::Unsigned(self.as_u64()),
            _ => IntOperand::Signed(self.as_i64()),
        }
    }
}

#[derive(Clone, Copy)]
enum IntOperand {
    Signed(i64),
    Unsigned(u64),
}

fn cmp_float_int(f: f64, i: IntOperand) -> Option<Ordering> {
    if f.is_nan() {
        return None;
    }
    let floor = f.floor();
    let result = match i {
        IntOperand::Signed(i) => {
            if floor < -(2.0f64.powi(63)) {
                Ordering::Greater
            } else if floor >= 2.0f64.powi(63) {
                Ordering::Less
            } else {
                refine_int_vs_float(i.cmp(&(floor as i64)), f, floor)
            }
        }
        IntOperand::Unsigned(i) => {
            if floor < 0.0 {
                Ordering::Greater
            } else if floor >= 2.0f64.powi(64) {
                Ordering::Less
            } else {
                refine_int_vs_float(i.cmp(&(floor as u64)), f, floor)
            }
        }
    };
    Some(result.reverse())
}

fn refine_int_vs_float(int_cmp_floor: Ordering, f: f64, floor: f64) -> Ordering {
    match int_cmp_floor {
        Ordering::Less => Ordering::Less,
        Ordering::Greater => Ordering::Greater,
        Ordering::Equal => {
            if f == floor {
                Ordering::Equal
            } else {
                Ordering::Less
            }
        }
    }
}

/// Errors raised by the scalar byte codec.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ScalarError {
    #[error("cannot encode or decode scalars with unspecified dtype")]
    UnspecifiedDtype,
    #[error("byte length {len} is not a multiple of element size {elem} for {dtype:?}")]
    ByteLengthMismatch {
        len: usize,
        elem: usize,
        dtype: DType,
    },
    #[error("value {value} is out of range for {dtype:?}")]
    OutOfRange { value: String, dtype: DType },
}

fn signed_dtype_range(dtype: DType) -> Option<(i64, i64)> {
    Some(match dtype {
        DType::Bool => (0, 1),
        DType::Uint8 => (0, u8::MAX as i64),
        DType::Int8 => (i8::MIN as i64, i8::MAX as i64),
        DType::Int16 => (i16::MIN as i64, i16::MAX as i64),
        DType::Uint16 => (0, u16::MAX as i64),
        DType::Int32 => (i32::MIN as i64, i32::MAX as i64),
        DType::Uint32 => (0, u32::MAX as i64),
        DType::Int64 => (i64::MIN, i64::MAX),
        DType::Uint64 => (0, i64::MAX),
        _ => return None,
    })
}

/// Reject a signed-integer bound that does not fit `dtype`'s exact integer
/// range.
pub fn check_int_in_dtype_range(value: i64, dtype: DType) -> Result<(), ScalarError> {
    if dtype == DType::Unspecified {
        return Err(ScalarError::UnspecifiedDtype);
    }
    if let Some((lo, hi)) = signed_dtype_range(dtype)
        && !(lo..=hi).contains(&value)
    {
        return Err(ScalarError::OutOfRange {
            value: value.to_string(),
            dtype,
        });
    }
    Ok(())
}

/// Reject an unsigned-integer bound that does not fit `dtype`'s exact integer
/// range.
pub fn check_uint_in_dtype_range(value: u64, dtype: DType) -> Result<(), ScalarError> {
    if dtype == DType::Unspecified {
        return Err(ScalarError::UnspecifiedDtype);
    }
    let out_of_range = match dtype {
        DType::Uint64 => false,
        DType::Bool => value > 1,
        DType::Uint8 => value > u8::MAX as u64,
        DType::Uint16 => value > u16::MAX as u64,
        DType::Uint32 => value > u32::MAX as u64,
        DType::Int8 => value > i8::MAX as u64,
        DType::Int16 => value > i16::MAX as u64,
        DType::Int32 => value > i32::MAX as u64,
        DType::Int64 => value > i64::MAX as u64,
        _ => false,
    };
    if out_of_range {
        return Err(ScalarError::OutOfRange {
            value: value.to_string(),
            dtype,
        });
    }
    Ok(())
}

/// Encode scalars as little-endian element bytes of `dtype`.
///
/// Booleans become 0/1 in any dtype; floats targeting `Uint8` use a saturating
/// cast, while floats targeting other integer dtypes saturate to `i64` and then
/// wrap to the target width; integers targeting float dtypes convert through
/// `f64`.
pub fn encode_scalars(values: &[Scalar], dtype: DType) -> Result<Vec<u8>, ScalarError> {
    if dtype == DType::Unspecified {
        return Err(ScalarError::UnspecifiedDtype);
    }
    let mut encoded = Vec::with_capacity(values.len() * dtype_size(dtype));
    for &value in values {
        encode_one(&mut encoded, value, dtype);
    }
    Ok(encoded)
}

/// Encode integers as little-endian element bytes of `dtype`.
pub fn encode_i64_scalars(values: &[i64], dtype: DType) -> Result<Vec<u8>, ScalarError> {
    if dtype == DType::Unspecified {
        return Err(ScalarError::UnspecifiedDtype);
    }
    let mut encoded = Vec::with_capacity(values.len() * dtype_size(dtype));
    for &value in values {
        encode_one(&mut encoded, Scalar::Int(value), dtype);
    }
    Ok(encoded)
}

fn encode_one(out: &mut Vec<u8>, value: Scalar, dtype: DType) {
    match dtype {
        DType::Bool => out.push(match value {
            Scalar::Bool(value) => u8::from(value),
            Scalar::Int(value) => u8::from(value != 0),
            Scalar::Float(value) => u8::from(value != 0.0),
        }),
        DType::Uint8 => out.push(match value {
            Scalar::Bool(value) => u8::from(value),
            Scalar::Int(value) => value as u8,
            Scalar::Float(value) => value as u8,
        }),
        DType::Int8 => out.extend_from_slice(&(value.as_i64() as i8).to_le_bytes()),
        DType::Int16 => out.extend_from_slice(&(value.as_i64() as i16).to_le_bytes()),
        DType::Int32 => out.extend_from_slice(&(value.as_i64() as i32).to_le_bytes()),
        DType::Int64 => out.extend_from_slice(&value.as_i64().to_le_bytes()),
        DType::Uint16 => out.extend_from_slice(&(value.as_i64() as u16).to_le_bytes()),
        DType::Uint32 => out.extend_from_slice(&(value.as_i64() as u32).to_le_bytes()),
        DType::Uint64 => out.extend_from_slice(&(value.as_i64() as u64).to_le_bytes()),
        DType::Float16 => out.extend_from_slice(&f64_to_f16_bits(float_value(value)).to_le_bytes()),
        DType::Float32 => out.extend_from_slice(&(float_value(value) as f32).to_le_bytes()),
        DType::Float64 => out.extend_from_slice(&float_value(value).to_le_bytes()),
        DType::Unspecified => unreachable!("checked by the public entry points"),
    }
}

fn float_value(value: Scalar) -> f64 {
    match value {
        Scalar::Bool(value) => f64::from(u8::from(value)),
        Scalar::Int(value) => value as f64,
        Scalar::Float(value) => value,
    }
}

/// Correctly round an `f64` to IEEE-754 binary16 bits (round to nearest, ties
/// to even), identically on every architecture.
pub fn f64_to_f16_bits(value: f64) -> u16 {
    // Round-to-odd through f32 avoids double rounding, matching a direct f64->f16 round as numpy does.
    // `half::f16::from_f64` double-rounds on x86 F16C and in its software fallback.
    let nearest = value as f32;
    let odd = if f64::from(nearest) == value {
        nearest
    } else {
        f32::from_bits(nearest.to_bits() | 1)
    };
    f16::from_f32(odd).to_bits()
}

/// Decode little-endian element bytes of `dtype` into scalars.
///
/// `u64` values above `i64::MAX` wrap, since [`Scalar`] has no unsigned variant;
/// read them back with [`Scalar::as_u64`] or [`Scalar::to_f64`].
///
/// # Errors
///
/// Trailing bytes that do not form a whole element.
pub fn decode_scalars(bytes: &[u8], dtype: DType) -> Result<Vec<Scalar>, ScalarError> {
    if dtype == DType::Unspecified {
        return Err(ScalarError::UnspecifiedDtype);
    }
    let elem = dtype_size(dtype);
    if !bytes.len().is_multiple_of(elem) {
        return Err(ScalarError::ByteLengthMismatch {
            len: bytes.len(),
            elem,
            dtype,
        });
    }
    Ok(bytes
        .chunks_exact(elem)
        .map(|chunk| decode_scalar(chunk, dtype))
        .collect())
}

/// Decode one little-endian element of `dtype` (`chunk` is exactly
/// `dtype_size(dtype)` bytes) with [`decode_scalars`]' semantics.
///
/// # Panics
///
/// On `Unspecified` or a chunk of the wrong length; callers check both.
pub(crate) fn decode_scalar(chunk: &[u8], dtype: DType) -> Scalar {
    match dtype {
        DType::Bool => Scalar::Bool(chunk[0] != 0),
        DType::Uint8 => Scalar::Int(chunk[0] as i64),
        DType::Int8 => Scalar::Int(chunk[0] as i8 as i64),
        DType::Int16 => Scalar::Int(i16::from_le_bytes(le_bytes(chunk)) as i64),
        DType::Uint16 => Scalar::Int(u16::from_le_bytes(le_bytes(chunk)) as i64),
        DType::Int32 => Scalar::Int(i32::from_le_bytes(le_bytes(chunk)) as i64),
        DType::Uint32 => Scalar::Int(u32::from_le_bytes(le_bytes(chunk)) as i64),
        DType::Int64 => Scalar::Int(i64::from_le_bytes(le_bytes(chunk))),
        DType::Uint64 => Scalar::Int(u64::from_le_bytes(le_bytes(chunk)) as i64),
        DType::Float16 => Scalar::Float(f16::from_le_bytes(le_bytes(chunk)).to_f64()),
        DType::Float32 => Scalar::Float(f32::from_le_bytes(le_bytes(chunk)) as f64),
        DType::Float64 => Scalar::Float(f64::from_le_bytes(le_bytes(chunk))),
        DType::Unspecified => unreachable!("decode_scalar called with an unspecified dtype"),
    }
}

fn le_bytes<const N: usize>(chunk: &[u8]) -> [u8; N] {
    chunk.try_into().expect("chunks_exact yields N-byte chunks")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_encoding_v1_float_golden() {
        let double_rounding = 1.0_f64 + 1.0 / 2048.0 + 1.0 / 33_554_432.0;
        let cases: &[(&str, f64, DType, &[u8])] = &[
            ("f16 +0.0", 0.0, DType::Float16, &[0x00, 0x00]),
            ("f16 -0.0", -0.0, DType::Float16, &[0x00, 0x80]),
            ("f16 1.0", 1.0, DType::Float16, &[0x00, 0x3C]),
            ("f16 +inf", f64::INFINITY, DType::Float16, &[0x00, 0x7C]),
            ("f16 -inf", f64::NEG_INFINITY, DType::Float16, &[0x00, 0xFC]),
            ("f16 max finite", 65504.0, DType::Float16, &[0xFF, 0x7B]),
            (
                "f16 min subnormal",
                1.0 / 16_777_216.0,
                DType::Float16,
                &[0x01, 0x00],
            ),
            ("f16 NaN(quiet)", f64::NAN, DType::Float16, &[0x00, 0x7E]),
            (
                "f16 double-rounding",
                double_rounding,
                DType::Float16,
                &[0x01, 0x3C],
            ),
            ("f32 1.0", 1.0, DType::Float32, &[0x00, 0x00, 0x80, 0x3F]),
            ("f32 -0.0", -0.0, DType::Float32, &[0x00, 0x00, 0x00, 0x80]),
            (
                "f64 1.0",
                1.0,
                DType::Float64,
                &[0, 0, 0, 0, 0, 0, 0xF0, 0x3F],
            ),
            (
                "f64 -0.0",
                -0.0,
                DType::Float64,
                &[0, 0, 0, 0, 0, 0, 0, 0x80],
            ),
        ];
        for (label, value, dtype, want) in cases {
            let got = encode_scalars(&[Scalar::Float(*value)], *dtype).expect("encode");
            assert_eq!(&got, want, "{label}: got {got:02x?}, want {want:02x?}");
        }
    }

    #[test]
    fn value_encoding_v1_integer_golden() {
        let cases: &[(&str, &[i64], DType, &[u8])] = &[
            (
                "i8",
                &[0, 1, i8::MIN as i64, i8::MAX as i64],
                DType::Int8,
                &[0x00, 0x01, 0x80, 0x7F],
            ),
            (
                "i16",
                &[0, 1, i16::MIN as i64, i16::MAX as i64],
                DType::Int16,
                &[0x00, 0x00, 0x01, 0x00, 0x00, 0x80, 0xFF, 0x7F],
            ),
            (
                "i32",
                &[0, 1, i32::MIN as i64, i32::MAX as i64],
                DType::Int32,
                &[
                    0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xFF,
                    0xFF, 0xFF, 0x7F,
                ],
            ),
            (
                "i64",
                &[0, 1, i64::MIN, i64::MAX],
                DType::Int64,
                &[
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xFF, 0xFF,
                    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F,
                ],
            ),
            (
                "u8",
                &[0, 1, u8::MIN as i64, u8::MAX as i64],
                DType::Uint8,
                &[0x00, 0x01, 0x00, 0xFF],
            ),
            (
                "u16",
                &[0, 1, u16::MIN as i64, u16::MAX as i64],
                DType::Uint16,
                &[0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xFF, 0xFF],
            ),
            (
                "u32",
                &[0, 1, u32::MIN as i64, u32::MAX as i64],
                DType::Uint32,
                &[
                    0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF,
                    0xFF, 0xFF, 0xFF,
                ],
            ),
            (
                "u64",
                &[0, 1, u64::MIN as i64, u64::MAX as i64],
                DType::Uint64,
                &[
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF,
                    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                ],
            ),
        ];
        for (label, values, dtype, want) in cases {
            let scalars: Vec<Scalar> = values.iter().copied().map(Scalar::Int).collect();
            let got = encode_scalars(&scalars, *dtype).expect("encode");
            assert_eq!(&got, want, "{label}: got {got:02x?}, want {want:02x?}");
        }
    }

    #[test]
    fn value_encoding_v1_decode_golden() {
        let cases: &[(&str, &[u8], DType, &[Scalar])] = &[
            (
                "f16",
                &[0x00, 0x3C, 0x00, 0x3E, 0x00, 0xC0],
                DType::Float16,
                &[Scalar::Float(1.0), Scalar::Float(1.5), Scalar::Float(-2.0)],
            ),
            (
                "f32",
                &[
                    0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0xC0, 0x3F, 0x00, 0x00, 0x00, 0xC0,
                ],
                DType::Float32,
                &[Scalar::Float(1.0), Scalar::Float(1.5), Scalar::Float(-2.0)],
            ),
            (
                "f64",
                &[
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0xF8, 0x3F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xC0,
                ],
                DType::Float64,
                &[Scalar::Float(1.0), Scalar::Float(1.5), Scalar::Float(-2.0)],
            ),
            (
                "i8",
                &[0x00, 0x01, 0x80, 0x7F],
                DType::Int8,
                &[
                    Scalar::Int(0),
                    Scalar::Int(1),
                    Scalar::Int(i8::MIN as i64),
                    Scalar::Int(i8::MAX as i64),
                ],
            ),
            (
                "i16",
                &[0x00, 0x00, 0x01, 0x00, 0x00, 0x80, 0xFF, 0x7F],
                DType::Int16,
                &[
                    Scalar::Int(0),
                    Scalar::Int(1),
                    Scalar::Int(i16::MIN as i64),
                    Scalar::Int(i16::MAX as i64),
                ],
            ),
            (
                "i32",
                &[
                    0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xFF,
                    0xFF, 0xFF, 0x7F,
                ],
                DType::Int32,
                &[
                    Scalar::Int(0),
                    Scalar::Int(1),
                    Scalar::Int(i32::MIN as i64),
                    Scalar::Int(i32::MAX as i64),
                ],
            ),
            (
                "i64",
                &[
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0xFF, 0xFF,
                    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F,
                ],
                DType::Int64,
                &[
                    Scalar::Int(0),
                    Scalar::Int(1),
                    Scalar::Int(i64::MIN),
                    Scalar::Int(i64::MAX),
                ],
            ),
            (
                "u8",
                &[0x00, 0x01, 0x00, 0xFF],
                DType::Uint8,
                &[
                    Scalar::Int(0),
                    Scalar::Int(1),
                    Scalar::Int(0),
                    Scalar::Int(u8::MAX as i64),
                ],
            ),
            (
                "u16",
                &[0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xFF, 0xFF],
                DType::Uint16,
                &[
                    Scalar::Int(0),
                    Scalar::Int(1),
                    Scalar::Int(0),
                    Scalar::Int(u16::MAX as i64),
                ],
            ),
            (
                "u32",
                &[
                    0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF,
                    0xFF, 0xFF, 0xFF,
                ],
                DType::Uint32,
                &[
                    Scalar::Int(0),
                    Scalar::Int(1),
                    Scalar::Int(0),
                    Scalar::Int(u32::MAX as i64),
                ],
            ),
            (
                "u64",
                &[
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF,
                    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                ],
                DType::Uint64,
                &[
                    Scalar::Int(0),
                    Scalar::Int(1),
                    Scalar::Int(0),
                    Scalar::Int(u64::MAX as i64),
                ],
            ),
        ];
        for (label, bytes, dtype, want) in cases {
            let got = decode_scalars(bytes, *dtype).expect("decode");
            assert_eq!(&got, want, "{label}: got {got:?}, want {want:?}");
        }
    }

    #[test]
    fn test_scalar_roundtrip_all_dtypes() {
        let cases: [(DType, Vec<Scalar>); 12] = [
            (DType::Bool, vec![Scalar::Bool(true), Scalar::Bool(false)]),
            (DType::Uint8, vec![Scalar::Int(0), Scalar::Int(255)]),
            (DType::Int8, vec![Scalar::Int(-128), Scalar::Int(127)]),
            (DType::Int16, vec![Scalar::Int(-5), Scalar::Int(999)]),
            (DType::Uint16, vec![Scalar::Int(0), Scalar::Int(65_535)]),
            (DType::Int32, vec![Scalar::Int(-5), Scalar::Int(70_000)]),
            (DType::Uint32, vec![Scalar::Int(0), Scalar::Int(70_000)]),
            (DType::Int64, vec![Scalar::Int(-5), Scalar::Int(1 << 40)]),
            (DType::Uint64, vec![Scalar::Int(0), Scalar::Int(1 << 40)]),
            (
                DType::Float16,
                vec![Scalar::Float(1.5), Scalar::Float(-2.0)],
            ),
            (
                DType::Float32,
                vec![Scalar::Float(1.5), Scalar::Float(-2.0)],
            ),
            (
                DType::Float64,
                vec![Scalar::Float(1.5), Scalar::Float(-2.0)],
            ),
        ];
        for (dtype, values) in cases {
            let bytes = encode_scalars(&values, dtype).expect("encode");
            assert_eq!(bytes.len(), values.len() * dtype_size(dtype), "{dtype:?}");
            let decoded = decode_scalars(&bytes, dtype).expect("decode");
            assert_eq!(decoded, values, "{dtype:?}");
        }
    }

    #[test]
    fn test_u64_above_i64_max_wraps() {
        let bytes = u64::MAX.to_le_bytes();
        let decoded = decode_scalars(&bytes, DType::Uint64).expect("decode");
        assert_eq!(decoded, vec![Scalar::Int(-1)]);
        let encoded = encode_scalars(&decoded, DType::Uint64).expect("encode");
        assert_eq!(encoded, bytes);
    }

    #[test]
    fn test_float_to_uint8_saturates_but_other_ints_wrap_via_i64() {
        let encoded = encode_scalars(&[Scalar::Float(300.5), Scalar::Float(-3.0)], DType::Uint8)
            .expect("encode");
        assert_eq!(encoded, vec![255, 0]);

        let encoded = encode_scalars(&[Scalar::Float(-3.0)], DType::Uint16).expect("encode");
        assert_eq!(encoded, (-3i64 as u16).to_le_bytes().to_vec());
    }

    #[test]
    fn test_int_to_float16_converts_through_f64() {
        let value = (1 << 24) + 1;
        let encoded = encode_scalars(&[Scalar::Int(value)], DType::Float16).expect("encode");
        assert_eq!(
            encoded,
            f64_to_f16_bits(value as f64).to_le_bytes().to_vec()
        );
    }

    #[test]
    fn f64_to_f16_bits_rounds_corners_correctly() {
        assert_eq!(f64_to_f16_bits(65504.0), 0x7BFF);
        assert_eq!(f64_to_f16_bits(65519.0), 0x7BFF);
        assert_eq!(f64_to_f16_bits(65520.0), 0x7C00);
        assert_eq!(f64_to_f16_bits(70000.0), 0x7C00);

        assert_eq!(f64_to_f16_bits(2.0f64.powi(-24)), 0x0001);
        assert_eq!(f64_to_f16_bits(2.0f64.powi(-25)), 0x0000);
        assert_eq!(f64_to_f16_bits(2.0f64.powi(-25) * 1.5), 0x0001);
        assert_eq!(f64_to_f16_bits(1023.0 * 2.0f64.powi(-24)), 0x03FF);
        assert_eq!(f64_to_f16_bits(2.0f64.powi(-14)), 0x0400);

        let sentinel = 1.0_f64 + 1.0 / 2048.0 + 1.0 / 33_554_432.0;
        assert_eq!(f64_to_f16_bits(sentinel), 0x3C01);
        assert_eq!(f64_to_f16_bits(-sentinel), 0xBC01);
        assert_eq!(f64_to_f16_bits(-2.0), 0xC000);

        assert_eq!(f64_to_f16_bits(2.0f64.powi(-30)), 0x0000);
        assert_eq!(f64_to_f16_bits(-0.0), 0x8000);
    }

    #[test]
    fn test_bool_coerces_into_every_dtype() {
        for dtype in DType::ALL {
            if dtype == DType::Unspecified {
                continue;
            }
            let encoded =
                encode_scalars(&[Scalar::Bool(true), Scalar::Bool(false)], dtype).expect("encode");
            let decoded = decode_scalars(&encoded, dtype).expect("decode");
            assert_eq!(decoded[0].as_i64(), 1, "{dtype:?}");
            assert_eq!(decoded[1].as_i64(), 0, "{dtype:?}");
        }
    }

    #[test]
    fn test_trailing_bytes_are_an_error() {
        let result = decode_scalars(&[0, 1, 2, 3, 4], DType::Float32);
        assert_eq!(
            result,
            Err(ScalarError::ByteLengthMismatch {
                len: 5,
                elem: 4,
                dtype: DType::Float32
            })
        );
    }

    #[test]
    fn test_unspecified_dtype_is_rejected() {
        assert_eq!(
            encode_scalars(&[Scalar::Int(1)], DType::Unspecified),
            Err(ScalarError::UnspecifiedDtype)
        );
        assert_eq!(
            decode_scalars(&[0, 0, 0, 0], DType::Unspecified),
            Err(ScalarError::UnspecifiedDtype)
        );
        assert_eq!(
            encode_i64_scalars(&[1], DType::Unspecified),
            Err(ScalarError::UnspecifiedDtype)
        );
    }

    #[test]
    fn test_cmp_typed_uint64_compares_unsigned() {
        let max = decode_scalars(&u64::MAX.to_le_bytes(), DType::Uint64).unwrap()[0];
        let zero = Scalar::Int(0);
        assert_eq!(max.cmp_typed(zero, DType::Uint64), Some(Ordering::Greater));
        assert_eq!(zero.cmp_typed(max, DType::Uint64), Some(Ordering::Less));
    }

    #[test]
    fn test_cmp_typed_mixed_float_int_is_exact_no_truncation() {
        assert_eq!(
            Scalar::Int(0).cmp_typed(Scalar::Float(0.5), DType::Int64),
            Some(Ordering::Less)
        );
        assert_eq!(
            Scalar::Float(-1.0).cmp_typed(Scalar::Int(0), DType::Uint64),
            Some(Ordering::Less),
            "float low -1.0 must be below uint value 0"
        );
        let umax = Scalar::Int(-1);
        assert_eq!(
            umax.cmp_typed(Scalar::Float(100.0), DType::Uint64),
            Some(Ordering::Greater)
        );
    }

    #[test]
    fn test_cmp_typed_float_int_edge_at_i64_extremes() {
        let f = i64::MAX as f64;
        assert_eq!(
            Scalar::Int(i64::MAX).cmp_typed(Scalar::Float(f), DType::Int64),
            Some(Ordering::Less)
        );
        let f = i64::MIN as f64;
        assert_eq!(
            Scalar::Int(i64::MIN).cmp_typed(Scalar::Float(f), DType::Int64),
            Some(Ordering::Equal)
        );
    }

    #[test]
    fn test_cmp_typed_float_int_fractional_and_infinity() {
        assert_eq!(
            Scalar::Int(3).cmp_typed(Scalar::Float(2.9), DType::Int64),
            Some(Ordering::Greater)
        );
        assert_eq!(
            Scalar::Int(2).cmp_typed(Scalar::Float(2.9), DType::Int64),
            Some(Ordering::Less)
        );
        assert_eq!(
            Scalar::Int(0).cmp_typed(Scalar::Float(f64::INFINITY), DType::Int64),
            Some(Ordering::Less)
        );
        assert_eq!(
            Scalar::Int(0).cmp_typed(Scalar::Float(f64::NEG_INFINITY), DType::Int64),
            Some(Ordering::Greater)
        );
        assert_eq!(
            Scalar::Int(0).cmp_typed(Scalar::Float(f64::NAN), DType::Int64),
            None
        );
    }

    #[test]
    fn test_to_f64_uint64_reinterprets() {
        let umax = decode_scalars(&u64::MAX.to_le_bytes(), DType::Uint64).unwrap()[0];
        assert_eq!(umax.to_f64(DType::Uint64), u64::MAX as f64);
        assert_eq!(umax.to_f64(DType::Int64), -1.0);
    }

    #[test]
    fn test_check_int_in_dtype_range() {
        assert!(check_int_in_dtype_range(300, DType::Int8).is_err());
        assert!(check_int_in_dtype_range(127, DType::Int8).is_ok());
        assert!(check_int_in_dtype_range(-1, DType::Uint8).is_err());
        assert!(check_int_in_dtype_range(i64::MAX, DType::Int64).is_ok());
        assert!(check_int_in_dtype_range(i64::MAX, DType::Float32).is_ok());
    }

    #[test]
    fn test_check_uint_in_dtype_range() {
        assert!(check_uint_in_dtype_range(1u64 << 33, DType::Uint32).is_err());
        assert!(check_uint_in_dtype_range(u32::MAX as u64, DType::Uint32).is_ok());
        assert!(check_uint_in_dtype_range(u64::MAX, DType::Uint64).is_ok());
        assert!(check_uint_in_dtype_range(1u64 << 63, DType::Int64).is_err());
    }

    #[test]
    fn test_as_i64_truncates_and_saturates() {
        assert_eq!(Scalar::Float(2.9).as_i64(), 2);
        assert_eq!(Scalar::Float(-2.9).as_i64(), -2);
        assert_eq!(Scalar::Float(f64::INFINITY).as_i64(), i64::MAX);
        assert_eq!(Scalar::Bool(true).as_i64(), 1);
    }

    mod proptests {
        use proptest::prelude::*;

        use super::*;

        fn exact_range(dtype: DType) -> std::ops::RangeInclusive<i64> {
            match dtype {
                DType::Bool => 0..=1,
                DType::Uint8 => 0..=u8::MAX as i64,
                DType::Int8 => i8::MIN as i64..=i8::MAX as i64,
                DType::Int16 => i16::MIN as i64..=i16::MAX as i64,
                DType::Uint16 => 0..=u16::MAX as i64,
                DType::Int32 => i32::MIN as i64..=i32::MAX as i64,
                DType::Uint32 => 0..=u32::MAX as i64,
                DType::Int64 | DType::Uint64 => i64::MIN..=i64::MAX,
                DType::Float16 => -2048..=2048,
                DType::Float32 => -(1 << 24)..=(1 << 24),
                DType::Float64 => -(1 << 53)..=(1 << 53),
                DType::Unspecified => unreachable!("not generated"),
            }
        }

        fn dtype_and_values() -> impl Strategy<Value = (DType, Vec<i64>)> {
            prop::sample::select(
                DType::ALL
                    .into_iter()
                    .filter(|&dtype| dtype != DType::Unspecified)
                    .collect::<Vec<_>>(),
            )
            .prop_flat_map(|dtype| {
                prop::collection::vec(exact_range(dtype), 0..=16)
                    .prop_map(move |values| (dtype, values))
            })
        }

        proptest! {
            #[test]
            fn prop_exact_integers_roundtrip((dtype, values) in dtype_and_values()) {
                let encoded = encode_i64_scalars(&values, dtype).expect("encode");
                prop_assert_eq!(encoded.len(), values.len() * dtype_size(dtype));
                let decoded = decode_scalars(&encoded, dtype).expect("decode");
                let roundtripped: Vec<i64> =
                    decoded.into_iter().map(Scalar::as_i64).collect();
                prop_assert_eq!(roundtripped, values);
            }
        }
    }
}
