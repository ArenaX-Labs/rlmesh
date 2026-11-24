//! Deserialization of the spec's non-negative integer fields (dim, count, index) with an error
//! that names the field's domain instead of the Rust type; the wire form stays a JSON integer.

use std::fmt;

use serde::Deserialize;
use serde::de::{self, Deserializer, Visitor};

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

    fn visit_f64<E: de::Error>(self, value: f64) -> Result<u32, E> {
        Err(E::custom(format!(
            "must be a non-negative integer, got {value}"
        )))
    }
}

pub(crate) fn de_count<'de, D: Deserializer<'de>>(deserializer: D) -> Result<u32, D::Error> {
    deserializer.deserialize_any(CountVisitor)
}

pub(crate) fn default_one() -> u32 {
    1
}

pub(crate) fn is_one(value: &u32) -> bool {
    *value == 1
}

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

struct Count(u32);

impl<'de> Deserialize<'de> for Count {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        de_count(deserializer).map(Count)
    }
}

pub(crate) fn de_opt_count<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<u32>, D::Error> {
    de_opt::<Count, D>(deserializer, "a non-negative integer or null")
        .map(|count| count.map(|Count(value)| value))
}

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

pub(crate) fn de_opt_number<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<f64>, D::Error> {
    de_opt::<Number, D>(deserializer, "a number or null")
        .map(|number| number.map(|Number(value)| value))
}

pub(crate) fn de_opt_numbers<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Vec<f64>>, D::Error> {
    let raw = Option::<Vec<Number>>::deserialize(deserializer)?;
    Ok(raw.map(|numbers| numbers.into_iter().map(|Number(value)| value).collect()))
}

pub(crate) const MAX_RENDER: u32 = 4096;

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
        let mut len = 2u32;
        while seq.next_element::<de::IgnoredAny>()?.is_some() {
            len += 1;
        }
        if len > 2 {
            return Err(de::Error::custom(format!(
                "range must be a pair of numbers [min, max], got {len}"
            )));
        }
        if pair.0 > pair.1 {
            return Err(de::Error::custom(format!(
                "range [min, max] must have min <= max, got [{}, {}]",
                pair.0, pair.1
            )));
        }
        Ok(pair)
    }
}

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
            fn visit_f64<E: de::Error>(self, value: f64) -> Result<i64, E> {
                Err(E::custom(format!("a whole number, got {value}")))
            }
        }
        deserializer.deserialize_any(DimVisitor).map(Dim)
    }
}

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

pub(crate) fn de_opt_dims<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Vec<i64>>, D::Error> {
    let raw = Option::<Vec<Dim>>::deserialize(deserializer)?;
    let Some(dims) = raw else { return Ok(None) };
    let dims: Vec<i64> = dims.into_iter().map(|Dim(value)| value).collect();
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
        let json = format!(r#"{{"role": "g", "dim": {}}}"#, super::MAX_DIM as u64 + 1);
        let err = serde_json::from_str::<Actuator>(&json).unwrap_err();
        assert!(err.to_string().contains("no larger than"), "got: {err}");
        let json = format!(r#"{{"role": "g", "dim": {}}}"#, super::MAX_DIM);
        let ok: Actuator = serde_json::from_str(&json).unwrap();
        assert_eq!(ok.dim, super::MAX_DIM);
    }
}
