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
///
/// ponytail: `1 << 24` is absurdly generous for any real spec (no image is 16M
/// px on a side, no state vector is 16M wide) yet stops an untrusted contract
/// from declaring a multi-GB dimension (e.g. `pad_to: 4_000_000_000` → a 16 GB
/// resize) or a product that integer-overflows `usize` in the apply path. It
/// bounds a single dimension, NOT a product: a spec within `MAX_DIM` can still
/// request a large allocation — a per-apply allocation budget is the upgrade
/// path if that ever bites.
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

/// `1` default for an "omitted-when-1" count field (`stack`, `execute_horizon`).
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
/// range error. Backs the `stack` and `execute_horizon` wire guards, which each
/// wrap it with their own field name and ceiling.
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

/// Deserialize an optional count (`Option<u32>`): `null` / absent -> `None`,
/// a present value through [`de_count`].
pub(crate) fn de_opt_count<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<u32>, D::Error> {
    struct OptCountVisitor;

    impl<'de> Visitor<'de> for OptCountVisitor {
        type Value = Option<u32>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a non-negative integer or null")
        }

        fn visit_none<E: de::Error>(self) -> Result<Option<u32>, E> {
            Ok(None)
        }

        fn visit_unit<E: de::Error>(self) -> Result<Option<u32>, E> {
            Ok(None)
        }

        fn visit_some<D: Deserializer<'de>>(
            self,
            deserializer: D,
        ) -> Result<Option<u32>, D::Error> {
            de_count(deserializer).map(Some)
        }
    }

    deserializer.deserialize_option(OptCountVisitor)
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
    struct OptNumberVisitor;

    impl<'de> Visitor<'de> for OptNumberVisitor {
        type Value = Option<f64>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a number or null")
        }

        fn visit_none<E: de::Error>(self) -> Result<Option<f64>, E> {
            Ok(None)
        }

        fn visit_unit<E: de::Error>(self) -> Result<Option<f64>, E> {
            Ok(None)
        }

        fn visit_some<D: Deserializer<'de>>(
            self,
            deserializer: D,
        ) -> Result<Option<f64>, D::Error> {
            Number::deserialize(deserializer).map(|Number(value)| Some(value))
        }
    }

    deserializer.deserialize_option(OptNumberVisitor)
}

struct RangeVisitor;

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
    struct OptRangeVisitor;

    impl<'de> Visitor<'de> for OptRangeVisitor {
        type Value = Option<(f64, f64)>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a range [min, max] or null")
        }

        fn visit_none<E: de::Error>(self) -> Result<Option<(f64, f64)>, E> {
            Ok(None)
        }

        fn visit_unit<E: de::Error>(self) -> Result<Option<(f64, f64)>, E> {
            Ok(None)
        }

        fn visit_some<D: Deserializer<'de>>(
            self,
            deserializer: D,
        ) -> Result<Option<(f64, f64)>, D::Error> {
            deserializer.deserialize_seq(RangeVisitor).map(Some)
        }
    }

    deserializer.deserialize_option(OptRangeVisitor)
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
