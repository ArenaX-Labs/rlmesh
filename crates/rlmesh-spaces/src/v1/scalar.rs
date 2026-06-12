use half::{bf16, f16};
use thiserror::Error;

use crate::v1::dtype::{DType, dtype_size};

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
}

/// Errors raised by the scalar byte codec.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ScalarError {
    #[error("cannot encode or decode scalars with unspecified dtype")]
    UnspecifiedDtype,
    #[error("byte length {len} is not a multiple of element size {elem} for {dtype:?}")]
    ByteLengthMismatch {
        len: usize,
        elem: usize,
        dtype: DType,
    },
}

/// Encode scalars as little-endian element bytes of `dtype`.
///
/// Cross-type coercions follow the wire codec's historical semantics:
/// booleans become 0/1 in any dtype; floats targeting `Uint8` use a direct
/// saturating cast, while floats targeting other integer dtypes saturate to
/// `i64` first and then wrap to the target width; integers targeting float
/// dtypes convert through `f64`.
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
        // Float -> Uint8 is a direct saturating cast (historical wire
        // behavior); every other integer dtype routes floats through i64.
        DType::Uint8 => out.push(match value {
            Scalar::Bool(value) => u8::from(value),
            Scalar::Int(value) => value as u8,
            Scalar::Float(value) => value as u8,
        }),
        DType::Int8 => out.extend_from_slice(&(int_value(value) as i8).to_le_bytes()),
        DType::Int16 => out.extend_from_slice(&(int_value(value) as i16).to_le_bytes()),
        DType::Int32 => out.extend_from_slice(&(int_value(value) as i32).to_le_bytes()),
        DType::Int64 => out.extend_from_slice(&int_value(value).to_le_bytes()),
        DType::Uint16 => out.extend_from_slice(&(int_value(value) as u16).to_le_bytes()),
        DType::Uint32 => out.extend_from_slice(&(int_value(value) as u32).to_le_bytes()),
        DType::Uint64 => out.extend_from_slice(&(int_value(value) as u64).to_le_bytes()),
        DType::Float16 => out.extend_from_slice(&f16::from_f64(float_value(value)).to_le_bytes()),
        DType::Bfloat16 => out.extend_from_slice(&bf16::from_f64(float_value(value)).to_le_bytes()),
        DType::Float32 => out.extend_from_slice(&(float_value(value) as f32).to_le_bytes()),
        DType::Float64 => out.extend_from_slice(&float_value(value).to_le_bytes()),
        DType::Unspecified => unreachable!("checked by the public entry points"),
    }
}

fn int_value(value: Scalar) -> i64 {
    value.as_i64()
}

fn float_value(value: Scalar) -> f64 {
    match value {
        Scalar::Bool(value) => {
            if value {
                1.0
            } else {
                0.0
            }
        }
        Scalar::Int(value) => value as f64,
        Scalar::Float(value) => value,
    }
}

/// Decode little-endian element bytes of `dtype` into scalars.
///
/// Strict: trailing bytes that do not form a whole element are an error.
/// `u64` values above `i64::MAX` wrap; [`Scalar`] has no unsigned variant.
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
    let decode_one = |chunk: &[u8]| -> Scalar {
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
            DType::Bfloat16 => Scalar::Float(bf16::from_le_bytes(le_bytes(chunk)).to_f64()),
            DType::Float32 => Scalar::Float(f32::from_le_bytes(le_bytes(chunk)) as f64),
            DType::Float64 => Scalar::Float(f64::from_le_bytes(le_bytes(chunk))),
            DType::Unspecified => unreachable!("checked above"),
        }
    };
    Ok(bytes.chunks_exact(elem).map(decode_one).collect())
}

fn le_bytes<const N: usize>(chunk: &[u8]) -> [u8; N] {
    chunk.try_into().expect("chunks_exact yields N-byte chunks")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_roundtrip_all_dtypes() {
        let cases: [(DType, Vec<Scalar>); 13] = [
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
                DType::Bfloat16,
                vec![Scalar::Float(1.0), Scalar::Float(-2.0)],
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
        // And wraps back on encode.
        let encoded = encode_scalars(&decoded, DType::Uint64).expect("encode");
        assert_eq!(encoded, bytes);
    }

    #[test]
    fn test_float_to_uint8_saturates_but_other_ints_wrap_via_i64() {
        // Historical wire semantics: Uint8 takes a direct saturating f64
        // cast; the other integer dtypes saturate to i64 then wrap.
        let encoded = encode_scalars(&[Scalar::Float(300.5), Scalar::Float(-3.0)], DType::Uint8)
            .expect("encode");
        assert_eq!(encoded, vec![255, 0]);

        let encoded = encode_scalars(&[Scalar::Float(-3.0)], DType::Uint16).expect("encode");
        assert_eq!(encoded, (-3i64 as u16).to_le_bytes().to_vec());
    }

    #[test]
    fn test_int_to_float16_converts_through_f64() {
        // 2^24 + 1 is not representable in f32; converting through f64
        // avoids double rounding before the f16 conversion.
        let value = (1 << 24) + 1;
        let encoded = encode_scalars(&[Scalar::Int(value)], DType::Float16).expect("encode");
        assert_eq!(encoded, f16::from_f64(value as f64).to_le_bytes().to_vec());
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
    fn test_as_i64_truncates_and_saturates() {
        assert_eq!(Scalar::Float(2.9).as_i64(), 2);
        assert_eq!(Scalar::Float(-2.9).as_i64(), -2);
        assert_eq!(Scalar::Float(f64::INFINITY).as_i64(), i64::MAX);
        assert_eq!(Scalar::Bool(true).as_i64(), 1);
    }

    mod proptests {
        use proptest::prelude::*;

        use super::*;

        /// The integer range each dtype represents exactly (floats are
        /// limited by their mantissa width).
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
                DType::Bfloat16 => -256..=256,
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
            /// Integers within a dtype's exact range survive the byte codec.
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
