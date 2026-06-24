//! Canonical leaf flatten / assemble over a `SpaceSpec`.
//!
//! The one sanctioned way to turn a composite [`SpaceValue`] into its ordered
//! fundamental leaves and back. Order is DFS pre-order: Dict children in
//! DECLARED key order (`DictSpec.keys`, **not** the value's `BTreeMap` order —
//! the guard against silently reordering leaves), Tuple by index, a fundamental
//! space is exactly one leaf. Decode pairs [`leaf_specs`] with the wire leaves;
//! encode pairs [`flatten_leaves`] with them; [`assemble_value`] rebuilds the
//! typed tree. See `value-transport-redesign.md` §3/§6 (WP1).

use std::collections::BTreeMap;

use super::SpaceValue;
use crate::errors::SpaceError;
use crate::types::{SpaceKind, SpaceSpec};

/// Fundamental leaf specs in canonical pre-order. Decode zips these with the
/// wire `leaves` to know each leaf's dtype/shape before assembling.
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
        // Box/Discrete/MultiBinary/MultiDiscrete/Text (and Unspecified) = one leaf.
        _ => out.push(spec),
    }
}

/// Leaf values of `value` in canonical order, driven by `spec`'s declared
/// structure. Errors on a structural mismatch (missing Dict key, wrong Tuple
/// arity, composite spec over a non-composite value).
///
/// ponytail: assumes each fundamental leaf already conforms to its spec — a leaf
/// kind mismatch (e.g. Box spec, Discrete value) is the per-leaf encoder's error,
/// not re-checked here. Run `conform` first if the value is untrusted.
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
        // fundamental leaf
        _ => {
            out.push(value);
            Ok(())
        }
    }
}

/// Rebuild a typed [`SpaceValue`] tree from per-leaf values in canonical order.
/// Errors if `leaves.len()` doesn't match the spec's leaf count (§3 hard
/// invariant — a wrong count must never silently truncate through `zip`).
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

    // Dict keys DELIBERATELY non-sorted (z before a): the builder can't produce
    // this, so it exercises the declared-order guard a raw/foreign DictSpec hits.
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
        // BTreeMap will store keys sorted (a, z); flatten must still follow the
        // spec's declared order (z, a).
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
        // Declared order (z, a), NOT BTreeMap order (a, z): z's Discrete(1) leads.
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
        // Too few leaves.
        assert!(assemble_value(&spec, vec![SpaceValue::Discrete(1)]).is_err());
        // Too many leaves.
        assert!(
            assemble_value(&spec, vec![SpaceValue::Discrete(0); 4]).is_err(),
            "extra leaf must not be silently dropped",
        );
        // Missing Dict key.
        let mut m = BTreeMap::new();
        m.insert("z".to_string(), SpaceValue::Discrete(1));
        assert!(flatten_leaves(&spec, &SpaceValue::Dict(m)).is_err());
    }
}
