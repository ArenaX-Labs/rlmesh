//! Domain-friendly deserialization for the non-negative integer (dim / count /
//! index) fields of the spec.
//!
//! serde's default `u32` error leaks the Rust wire type — a negative `dim`
//! reads `invalid value: integer `-1`, expected u32`. A spec author should not
//! see `u32`; these guards emit `must be a non-negative integer, got -1`
//! instead, while leaving the wire format unchanged (still a JSON integer).

use std::fmt;

use serde::Deserialize;
use serde::de::{self, Deserializer, Visitor};

/// Shared upper bound on every count/dim field on the wire (`dim`, `index`,
/// `pad_to`, `lead_dims`, image `height`/`width`; `stack` layers a tighter
/// `1..=64` on top via [`de_stack`](crate::spec::model::image)).
pub(crate) const MAX_DIM: u32 = 1 << 24;

struct CountVisitor;

impl<'de> Visitor<'de> for CountVisitor {
    type Value = u32;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a non-negative integer")
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<u32, E> {
        if value > u64::from(MAX_DIM) {
            return Err(E::custom(format!(
                "must be a non-negative integer no larger than {MAX_DIM}, got {value}"
            )));
        }
        Ok(value as u32)
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<u32, E> {
        if value < 0 {
            return Err(E::custom(format!(
                "must be a non-negative integer, got {value}"
            )));
        }
        self.visit_u64(value as u64)
    }

    // A count is an integer on the wire; a float literal (even whole-valued
    // like `3.0`) is rejected in domain language rather than leaking serde's
    // "floating point" wire phrasing.
    fn visit_f64<E: de::Error>(self, value: f64) -> Result<u32, E> {
        Err(E::custom(format!(
            "must be a non-negative integer, got {value}"
        )))
    }
}

/// Deserialize a required count (`u32`) with a domain-friendly error.
pub(crate) fn de_count<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u32, D::Error> {
    deserializer.deserialize_any(CountVisitor)
}

/// `1` default for an "omitted-when-1" count field (`stack`).
pub(crate) fn default_one() -> u32 {
    1
}

/// True when an "omitted-when-1" count field holds its default — the
/// `skip_serializing_if` that keeps a non-chunking/non-stacking layout
/// byte-identical with the Python serializer.
pub(crate) fn is_one(value: &u32) -> bool {
    *value == 1
}

/// Deserialize a count constrained to `1..=max`, routed through [`de_count`] (so
/// a negative/non-integer still reads in domain language) with a field-named
/// range error. Backs the `stack` wire guard, which wraps it with its own field
/// name and ceiling.
pub(crate) fn de_bounded_count<'de, D: Deserializer<'de>>(
    deserializer: D,
    field: &str,
    max: u32,
) -> Result<u32, D::Error> {
    let value = de_count(deserializer)?;
    if !(1..=max).contains(&value) {
        return Err(de::Error::custom(format!(
            "{field} must be between 1 and {max}, got {value}"
        )));
    }
    Ok(value)
}

/// Shared optional-field plumbing behind the `de_opt_*` deserializers: `null` /
/// absent -> `None`, a present value through `T::deserialize`. Each caller
/// keeps its own domain-language `expecting` string.
fn de_opt<'de, T: Deserialize<'de>, D: Deserializer<'de>>(
    deserializer: D,
    expecting: &'static str,
) -> Result<Option<T>, D::Error> {
    struct OptVisitor<T> {
        expecting: &'static str,
        marker: std::marker::PhantomData<T>,
    }

    impl<'de, T: Deserialize<'de>> Visitor<'de> for OptVisitor<T> {
        type Value = Option<T>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str(self.expecting)
        }

        fn visit_none<E: de::Error>(self) -> Result<Option<T>, E> {
            Ok(None)
        }

        fn visit_unit<E: de::Error>(self) -> Result<Option<T>, E> {
            Ok(None)
        }

        fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Option<T>, D::Error> {
            T::deserialize(deserializer).map(Some)
        }
    }

    deserializer.deserialize_option(OptVisitor {
        expecting,
        marker: std::marker::PhantomData,
    })
}

/// A count routed through [`de_count`], as a `Deserialize` impl so [`de_opt`]
/// can drive it.
struct Count(u32);

impl<'de> Deserialize<'de> for Count {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        de_count(deserializer).map(Count)
    }
}

/// Deserialize an optional count (`Option<u32>`): `null` / absent -> `None`,
/// a present value through [`de_count`].
pub(crate) fn de_opt_count<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<u32>, D::Error> {
    de_opt::<Count, D>(deserializer, "a non-negative integer or null")
        .map(|count| count.map(|Count(value)| value))
}

/// A single JSON number with a domain-friendly type error. serde's default
/// leaks the Rust wire type (`expected f64`); this reports `a number` and
/// widens ints to `f64`. Non-finite literals are already rejected by
/// serde_json at parse, so no extra finiteness check is needed here.
struct Number(f64);

impl<'de> Deserialize<'de> for Number {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct NumberVisitor;
        impl<'de> Visitor<'de> for NumberVisitor {
            type Value = f64;
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a number")
            }
            fn visit_f64<E: de::Error>(self, value: f64) -> Result<f64, E> {
                Ok(value)
            }
            fn visit_i64<E: de::Error>(self, value: i64) -> Result<f64, E> {
                Ok(value as f64)
            }
            fn visit_u64<E: de::Error>(self, value: u64) -> Result<f64, E> {
                Ok(value as f64)
            }
        }
        deserializer.deserialize_any(NumberVisitor).map(Number)
    }
}

/// Deserialize an optional single number (`Option<f64>`): `null`/absent → `None`,
/// a present value through [`Number`]. Used for `scale`/`threshold` so a
/// wrong-typed value reads `expected a number` instead of leaking serde's bare
/// `f64` — the same domain-language contract the sibling count/range fields keep.
pub(crate) fn de_opt_number<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<f64>, D::Error> {
    de_opt::<Number, D>(deserializer, "a number or null")
        .map(|number| number.map(|Number(value)| value))
}

/// Deserialize an optional list of numbers (`Option<Vec<f64>>`): the per-axis
/// `axis_scale` / `axis_offset` / `axis_fill` vectors. Each element goes
/// through [`Number`] so a wrong-typed entry reads `a number`; the list's
/// length is checked against the leaf's resolved width by the resolver, not
/// here (a `ConcatPart` learns its width only at resolve).
pub(crate) fn de_opt_numbers<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Vec<f64>>, D::Error> {
    let raw = Option::<Vec<Number>>::deserialize(deserializer)?;
    Ok(raw.map(|numbers| numbers.into_iter().map(|Number(value)| value).collect()))
}

/// Upper bound on each axis of a declared render size (the image `render`
/// assertion). A camera dial the platform binds from an untrusted spec, so the
/// ceiling is a sane display resolution rather than the shared [`MAX_DIM`].
pub(crate) const MAX_RENDER: u32 = 4096;

/// Deserialize an optional `[height, width]` size pair, with a bare integer as
/// the square shorthand (`448` == `[448, 448]`). Each axis is constrained to
/// `1..=`[`MAX_RENDER`]: a zero axis names no camera and an unbounded one would
/// have the env allocate an arbitrarily large frame. Backs the image `render`
/// assertion; emitted on the wire as the `[height, width]` pair.
pub(crate) fn de_opt_count_pair<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<(u32, u32)>, D::Error> {
    struct Pair((u32, u32));

    impl<'de> Deserialize<'de> for Pair {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            struct PairVisitor;

            impl<'de> Visitor<'de> for PairVisitor {
                type Value = (u32, u32);

                fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("a size, an integer or a pair [height, width]")
                }

                // A bare integer is the square shorthand; the count visitor keeps
                // the negative/float cases in domain language.
                fn visit_u64<E: de::Error>(self, value: u64) -> Result<(u32, u32), E> {
                    CountVisitor.visit_u64(value).map(|side| (side, side))
                }

                fn visit_i64<E: de::Error>(self, value: i64) -> Result<(u32, u32), E> {
                    CountVisitor.visit_i64(value).map(|side| (side, side))
                }

                fn visit_f64<E: de::Error>(self, value: f64) -> Result<(u32, u32), E> {
                    CountVisitor.visit_f64(value).map(|side| (side, side))
                }

                fn visit_seq<A: de::SeqAccess<'de>>(
                    self,
                    mut seq: A,
                ) -> Result<(u32, u32), A::Error> {
                    let mut axes: Vec<u32> = Vec::new();
                    while let Some(Count(axis)) = seq.next_element::<Count>()? {
                        axes.push(axis);
                    }
                    match axes[..] {
                        [height, width] => Ok((height, width)),
                        _ => Err(de::Error::custom(format!(
                            "a size is an integer or a pair [height, width], got {} element(s)",
                            axes.len()
                        ))),
                    }
                }
            }

            deserializer.deserialize_any(PairVisitor).map(Pair)
        }
    }

    let pair = de_opt::<Pair, D>(
        deserializer,
        "a size, an integer or a pair [height, width], or null",
    )
    .map(|pair| pair.map(|Pair(value)| value))?;
    if let Some((height, width)) = pair {
        for (axis, value) in [("height", height), ("width", width)] {
            if !(1..=MAX_RENDER).contains(&value) {
                return Err(de::Error::custom(format!(
                    "{axis} must be between 1 and {MAX_RENDER}, got {value}"
                )));
            }
        }
    }
    Ok(pair)
}

/// Deserialize an optional JPEG quality (`Option<u8>`) on the IJG `1..=100`
/// scale, routed through [`de_count`] so a negative or float literal still
/// reads in domain language. `0` is not a quality and anything past `100` is
/// off the scale the encoder is written on, so both are rejected at the wire
/// boundary rather than surfacing as a clamp nobody declared.
pub(crate) fn de_opt_jpeg_quality<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<u8>, D::Error> {
    let quality = de_opt::<Count, D>(deserializer, "a jpeg quality in 1..=100 or null")
        .map(|count| count.map(|Count(value)| value))?;
    if let Some(value) = quality
        && !(1..=100).contains(&value)
    {
        return Err(de::Error::custom(format!(
            "jpeg_quality must be between 1 and 100, got {value}"
        )));
    }
    Ok(quality.map(|value| value as u8))
}

/// Deserialize an optional fraction in `(0, 1]` (the image `crop` /
/// `crop_area` box), routed through [`Number`] so a wrong-typed value still
/// reads `a number`. `0` (keep nothing) and anything past the whole frame are
/// rejected at the wire boundary rather than silently producing an empty or
/// out-of-bounds crop.
pub(crate) fn de_opt_unit_fraction<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<f64>, D::Error> {
    let fraction = de_opt::<Number, D>(deserializer, "a fraction in (0, 1] or null")
        .map(|number| number.map(|Number(value)| value))?;
    if let Some(value) = fraction
        && !(value > 0.0 && value <= 1.0)
    {
        return Err(de::Error::custom(format!(
            "must be a fraction in (0, 1], got {value}"
        )));
    }
    Ok(fraction)
}

pub(crate) struct RangeVisitor;

impl<'de> Visitor<'de> for RangeVisitor {
    type Value = (f64, f64);

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a range, a pair of numbers [min, max]")
    }

    fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<(f64, f64), A::Error> {
        let pair = match (seq.next_element::<Number>()?, seq.next_element::<Number>()?) {
            (Some(min), Some(max)) => (min.0, max.0),
            (Some(_), None) => {
                return Err(de::Error::custom(
                    "range must be a pair of numbers [min, max], got 1",
                ));
            }
            _ => {
                return Err(de::Error::custom(
                    "range must be a pair of numbers [min, max], got 0",
                ));
            }
        };
        // Count any extra elements so the error reports the real length.
        let mut len = 2u32;
        while seq.next_element::<de::IgnoredAny>()?.is_some() {
            len += 1;
        }
        if len > 2 {
            return Err(de::Error::custom(format!(
                "range must be a pair of numbers [min, max], got {len}"
            )));
        }
        // A reversed range silently inverts whatever it scales (e.g. a
        // normalize range `[1, 0]` flips pixel polarity); reject it at the wire
        // boundary. `min == max` (a degenerate constant range) and unbounded
        // `±inf` bounds are left to the consumer.
        if pair.0 > pair.1 {
            return Err(de::Error::custom(format!(
                "range [min, max] must have min <= max, got [{}, {}]",
                pair.0, pair.1
            )));
        }
        Ok(pair)
    }
}

/// Deserialize an optional `[min, max]` range with domain-friendly errors.
/// serde's default tuple deserializer leaks `f64` and `tuple of size 2`; this
/// reports `a range, a pair of numbers [min, max]` and the real length.
pub(crate) fn de_opt_range<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<(f64, f64)>, D::Error> {
    struct Range((f64, f64));

    impl<'de> Deserialize<'de> for Range {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            deserializer.deserialize_seq(RangeVisitor).map(Range)
        }
    }

    de_opt::<Range, D>(deserializer, "a range [min, max] or null")
        .map(|range| range.map(|Range(pair)| pair))
}

/// A single JSON integer (signed; `-1` = "infer") with a domain-friendly type
/// error. serde's default leaks `i64`; this reports `a whole number`.
struct Dim(i64);

impl<'de> Deserialize<'de> for Dim {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct DimVisitor;
        impl<'de> Visitor<'de> for DimVisitor {
            type Value = i64;
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a whole number")
            }
            fn visit_i64<E: de::Error>(self, value: i64) -> Result<i64, E> {
                Ok(value)
            }
            fn visit_u64<E: de::Error>(self, value: u64) -> Result<i64, E> {
                i64::try_from(value).map_err(|_| {
                    E::custom(format!(
                        "a whole number no larger than {}, got {value}",
                        i64::MAX
                    ))
                })
            }
            // A reshape element is a whole number on the wire; reject a float
            // literal in domain language, not serde's "floating point" phrasing.
            fn visit_f64<E: de::Error>(self, value: f64) -> Result<i64, E> {
                Err(E::custom(format!("a whole number, got {value}")))
            }
        }
        deserializer.deserialize_any(DimVisitor).map(Dim)
    }
}

/// Deserialize the optional frame-history `offsets` list.
///
/// Elements go through [`Dim`] so a wrong-typed entry reads `a whole number`
/// instead of leaking `i32`. The window *law* — non-positive, strictly
/// increasing, ending at `0`, `len == stack`, and a span within the ceiling —
/// is a resolve check, not a codec one: like `resample`'s vocabulary, a list a
/// newer core understands parses here and fails with a typed resolve error
/// there rather than at the wire door.
pub(crate) fn de_offsets<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Vec<i32>>, D::Error> {
    let raw = Option::<Vec<Dim>>::deserialize(deserializer)?;
    let Some(offsets) = raw else { return Ok(None) };
    offsets
        .into_iter()
        .map(|Dim(value)| {
            i32::try_from(value).map_err(|_| {
                de::Error::custom(format!("a frame offset must fit in 32 bits, got {value}"))
            })
        })
        .collect::<Result<Vec<i32>, D::Error>>()
        .map(Some)
}

/// Deserialize an optional reshape spec (a list of dimensions, `-1` = infer)
/// with a domain-friendly element error instead of serde's `expected i64`.
pub(crate) fn de_opt_dims<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Vec<i64>>, D::Error> {
    let raw = Option::<Vec<Dim>>::deserialize(deserializer)?;
    let Some(dims) = raw else { return Ok(None) };
    let dims: Vec<i64> = dims.into_iter().map(|Dim(value)| value).collect();
    // A reshape element is a concrete size (>= 0) or a single `-1` (infer).
    // Reject the structurally-invalid cases (any other negative, or more than
    // one infer) here at the publish/normalize door, so a bad spec fails at
    // construction instead of per-step in apply. The length-dependent checks
    // (product == element count, infer divisibility) need the runtime value, so
    // they stay in apply.
    let mut infer = 0;
    for &value in &dims {
        if value < -1 {
            return Err(de::Error::custom(format!(
                "a reshape dimension is -1 (infer) or a non-negative size, got {value}"
            )));
        }
        if value == -1 {
            infer += 1;
            if infer > 1 {
                return Err(de::Error::custom(
                    "reshape allows at most one -1 (infer) dimension",
                ));
            }
        }
    }
    Ok(Some(dims))
}

#[cfg(test)]
mod tests {
    use crate::spec::Actuator;

    #[test]
    fn negative_count_reads_in_domain_language() {
        let err = serde_json::from_str::<Actuator>(r#"{"role": "g", "dim": -1}"#).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("must be a non-negative integer, got -1"),
            "got: {message}"
        );
        assert!(!message.contains("u32"), "leaks the wire type: {message}");
    }

    #[test]
    fn wrong_type_reads_in_domain_language() {
        let err = serde_json::from_str::<Actuator>(r#"{"role": "g", "dim": "x"}"#).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("non-negative integer"), "got: {message}");
        assert!(!message.contains("u32"), "leaks the wire type: {message}");
    }

    #[test]
    fn float_count_reads_in_domain_language() {
        // An integer-valued float literal is still rejected, but in domain
        // language — not serde's leaked "floating point" wire phrasing.
        let err = serde_json::from_str::<Actuator>(r#"{"role": "g", "dim": 3.0}"#).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("non-negative integer"), "got: {message}");
        assert!(
            !message.contains("floating point"),
            "leaks the wire phrasing: {message}"
        );
    }

    #[test]
    fn count_above_max_dim_is_rejected() {
        // The shared wire ceiling stops an untrusted spec from declaring a
        // dimension large enough to OOM/overflow the apply path.
        let json = format!(r#"{{"role": "g", "dim": {}}}"#, super::MAX_DIM as u64 + 1);
        let err = serde_json::from_str::<Actuator>(&json).unwrap_err();
        assert!(err.to_string().contains("no larger than"), "got: {err}");
        // The bound itself is accepted.
        let json = format!(r#"{{"role": "g", "dim": {}}}"#, super::MAX_DIM);
        let ok: Actuator = serde_json::from_str(&json).unwrap();
        assert_eq!(ok.dim, super::MAX_DIM);
    }
}
