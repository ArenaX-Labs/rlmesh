//! Canonical leaf flatten / assemble over a `SpaceSpec`.
//!
//! Order is depth-first pre-order: Dict children in declared key order (`DictSpec.keys`, not
//! the value's map order), Tuple by index, and a fundamental space is one leaf. Decode pairs
//! [`leaf_specs`] with the wire leaves, encode pairs [`flatten_leaves`] with them, and
//! [`assemble_value`] rebuilds the typed tree.

use std::collections::BTreeMap;

use super::SpaceValue;
use crate::errors::SpaceError;
use crate::types::{SpaceKind, SpaceSpec};

/// Fundamental leaf specs in canonical pre-order.
#[must_use]
pub fn leaf_specs(spec: &SpaceSpec) -> Vec<&SpaceSpec> {
    let mut out = Vec::new();
    collect_specs(spec, &mut out);
    out
}

fn collect_specs<'a>(spec: &'a SpaceSpec, out: &mut Vec<&'a SpaceSpec>) {
    match &spec.spec {
        Some(SpaceKind::Dict(d)) => d.spaces.iter().for_each(|s| collect_specs(s, out)),
        Some(SpaceKind::Tuple(t)) => t.spaces.iter().for_each(|s| collect_specs(s, out)),
        _ => out.push(spec),
    }
}

/// Leaf values of `value` in canonical order, driven by `spec`'s declared
/// structure. Dict children follow the declared key order, never the value's
/// `BTreeMap` order.
pub fn flatten_leaves<'v>(
    spec: &SpaceSpec,
    value: &'v SpaceValue,
) -> Result<Vec<&'v SpaceValue>, SpaceError> {
    let mut out = Vec::new();
    flatten_into(spec, value, &mut out)?;
    Ok(out)
}

fn flatten_into<'v>(
    spec: &SpaceSpec,
    value: &'v SpaceValue,
    out: &mut Vec<&'v SpaceValue>,
) -> Result<(), SpaceError> {
    match (&spec.spec, value) {
        (Some(SpaceKind::Dict(d)), SpaceValue::Dict(m)) => {
            for (key, child) in d.keys.iter().zip(&d.spaces) {
                let cv = m
                    .get(key)
                    .ok_or_else(|| SpaceError::invalid("$", format!("dict missing key {key:?}")))?;
                flatten_into(child, cv, out)?;
            }
            Ok(())
        }
        (Some(SpaceKind::Tuple(t)), SpaceValue::Tuple(v)) => {
            if v.len() != t.spaces.len() {
                return Err(SpaceError::invalid(
                    "$",
                    format!("tuple arity {} != spec {}", v.len(), t.spaces.len()),
                ));
            }
            for (child, cv) in t.spaces.iter().zip(v) {
                flatten_into(child, cv, out)?;
            }
            Ok(())
        }
        (Some(SpaceKind::Dict(_) | SpaceKind::Tuple(_)), _) => Err(SpaceError::invalid(
            "$",
            "composite spec but value is not the matching composite",
        )),
        _ => {
            out.push(value);
            Ok(())
        }
    }
}

/// Rebuild a typed [`SpaceValue`] tree from per-leaf values in canonical order.
///
/// # Errors
/// [`SpaceError`] when `leaves.len()` is not the spec's leaf count. Leaf values are not
/// checked against their specs.
pub fn assemble_value(spec: &SpaceSpec, leaves: Vec<SpaceValue>) -> Result<SpaceValue, SpaceError> {
    let mut it = leaves.into_iter();
    let value = assemble_from(spec, &mut it)?;
    if it.next().is_some() {
        return Err(SpaceError::invalid("$", "more leaves than spec expects"));
    }
    Ok(value)
}

fn assemble_from(
    spec: &SpaceSpec,
    leaves: &mut impl Iterator<Item = SpaceValue>,
) -> Result<SpaceValue, SpaceError> {
    match &spec.spec {
        Some(SpaceKind::Dict(d)) => {
            let mut m = BTreeMap::new();
            for (key, child) in d.keys.iter().zip(&d.spaces) {
                m.insert(key.clone(), assemble_from(child, leaves)?);
            }
            Ok(SpaceValue::Dict(m))
        }
        Some(SpaceKind::Tuple(t)) => {
            let mut v = Vec::with_capacity(t.spaces.len());
            for child in &t.spaces {
                v.push(assemble_from(child, leaves)?);
            }
            Ok(SpaceValue::Tuple(v))
        }
        _ => leaves
            .next()
            .ok_or_else(|| SpaceError::invalid("$", "fewer leaves than spec expects")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DictSpec, DiscreteSpec, TupleSpec};

    fn discrete() -> SpaceSpec {
        SpaceSpec {
            spec: Some(SpaceKind::Discrete(DiscreteSpec { n: 4, start: 0 })),
            ..Default::default()
        }
    }
    fn text() -> SpaceSpec {
        SpaceSpec {
            spec: Some(SpaceKind::Text(crate::types::TextSpec::default())),
            ..Default::default()
        }
    }

    fn nested_spec() -> SpaceSpec {
        SpaceSpec {
            spec: Some(SpaceKind::Dict(DictSpec {
                keys: vec!["z".into(), "a".into()],
                spaces: vec![
                    discrete(),
                    SpaceSpec {
                        spec: Some(SpaceKind::Tuple(TupleSpec {
                            spaces: vec![discrete(), text()],
                        })),
                        ..Default::default()
                    },
                ],
            })),
            ..Default::default()
        }
    }

    fn nested_value() -> SpaceValue {
        let mut m = BTreeMap::new();
        m.insert("z".to_string(), SpaceValue::Discrete(1));
        m.insert(
            "a".to_string(),
            SpaceValue::Tuple(vec![SpaceValue::Discrete(2), SpaceValue::Text("x".into())]),
        );
        SpaceValue::Dict(m)
    }

    #[test]
    fn declared_order_roundtrip() {
        let spec = nested_spec();
        let value = nested_value();

        assert_eq!(leaf_specs(&spec).len(), 3);

        let leaves = flatten_leaves(&spec, &value).unwrap();
        assert_eq!(
            leaves,
            vec![
                &SpaceValue::Discrete(1),
                &SpaceValue::Discrete(2),
                &SpaceValue::Text("x".into()),
            ]
        );

        let owned: Vec<SpaceValue> = leaves.into_iter().cloned().collect();
        assert_eq!(assemble_value(&spec, owned).unwrap(), value);
    }

    #[test]
    fn count_and_structure_mismatches_error() {
        let spec = nested_spec();
        assert!(assemble_value(&spec, vec![SpaceValue::Discrete(1)]).is_err());
        assert!(
            assemble_value(&spec, vec![SpaceValue::Discrete(0); 4]).is_err(),
            "extra leaf must not be silently dropped",
        );
        let mut m = BTreeMap::new();
        m.insert("z".to_string(), SpaceValue::Discrete(1));
        assert!(flatten_leaves(&spec, &SpaceValue::Dict(m)).is_err());
    }
}
