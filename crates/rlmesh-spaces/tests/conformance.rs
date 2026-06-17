//! Value-conformance contract suite for the `2026.06` workflow edition.
//!
//! Each case pins one clause of the edition's value-conformance contract: whether
//! a value is a full member of its space (`Member`), a **structural** deviation
//! (always rejected, regardless of the serving side's validation policy), or a
//! **range** deviation (tolerated under `warn`, rejected under `strict`).
//!
//! The cases are *data*, not bespoke test functions, and the harness only depends
//! on the public conformance API (`conform` / `contains`). That is deliberate:
//! when a second edition mints and changes a behavior, this table can be
//! re-pointed or forked per edition without rewriting the harness. The
//! edition-selection plumbing does not exist yet — the contract is currently
//! fixed at `2026.06` — but the suite is shaped so moving it behind an edition
//! parameter later is a mechanical change, not a rewrite.
//!
//! Scope: this file covers the structural/range *classification* the spaces crate
//! owns. The policy application (warn/strict/off + the conformance-warning
//! channel) and dtype coercion live in the server and codec crates and carry
//! their own conformance cases; together they form the full 2026.06 suite.

use std::collections::BTreeMap;

use rlmesh_spaces::spaces::{
    BoxSpaceBuilder, Conformance, DictSpaceBuilder, DiscreteBuilder, MultiBinaryBuilder,
    MultiDiscreteBuilder, PolicyOutcome, SpaceSpec, SpaceValue, TextBuilder, TupleSpaceBuilder,
    ValidationPolicy, conform, contains,
};
use rlmesh_spaces::{DType, Tensor};

/// Expected classification of a value against a space under 2026.06.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Expect {
    /// Full member: structural and range conformance both hold.
    Member,
    /// Structural deviation: rejected regardless of validation policy.
    Structural,
    /// Range deviation: tolerated under `warn`, rejected under `strict`.
    Range,
}

struct Case {
    clause: &'static str,
    space: SpaceSpec,
    value: SpaceValue,
    expect: Expect,
}

fn f32_box(low: f64, high: f64, shape: Vec<i64>) -> SpaceSpec {
    BoxSpaceBuilder::scalar(low, high, shape)
        .dtype(DType::Float32)
        .build()
        .expect("valid space")
}

fn f32_tensor(values: &[f32], shape: Vec<i64>) -> SpaceValue {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    SpaceValue::Box(Tensor::from_vec(bytes, shape, DType::Float32).expect("valid tensor"))
}

fn cases() -> Vec<Case> {
    let unit_box = || f32_box(0.0, 1.0, vec![1]);

    vec![
        // ----- Members: structural and range both hold -----
        Case {
            clause: "box within bounds",
            space: f32_box(0.0, 1.0, vec![2]),
            value: f32_tensor(&[0.0, 1.0], vec![2]),
            expect: Expect::Member,
        },
        Case {
            clause: "box accepts +inf against an infinite bound",
            space: f32_box(f64::NEG_INFINITY, f64::INFINITY, vec![1]),
            value: f32_tensor(&[f32::INFINITY], vec![1]),
            expect: Expect::Member,
        },
        Case {
            clause: "discrete within domain",
            space: DiscreteBuilder::new(4).build().expect("valid test fixture"),
            value: SpaceValue::Discrete(3),
            expect: Expect::Member,
        },
        Case {
            clause: "multidiscrete within per-element domain",
            space: MultiDiscreteBuilder::vector([2, 3])
                .build()
                .expect("valid test fixture"),
            value: SpaceValue::MultiDiscrete(vec![1, 2]),
            expect: Expect::Member,
        },
        Case {
            clause: "multibinary correct shape",
            space: MultiBinaryBuilder::scalar(3)
                .build()
                .expect("valid test fixture"),
            value: SpaceValue::MultiBinary(vec![true, false, true]),
            expect: Expect::Member,
        },
        Case {
            clause: "text within length and charset",
            space: TextBuilder::new(5)
                .min_length(1)
                .charset("abc")
                .build()
                .expect("valid test fixture"),
            value: SpaceValue::Text("abc".into()),
            expect: Expect::Member,
        },
        Case {
            clause: "dict with all children valid",
            space: DictSpaceBuilder::new()
                .insert("a", unit_box())
                .insert(
                    "b",
                    DiscreteBuilder::new(3).build().expect("valid test fixture"),
                )
                .build()
                .expect("valid test fixture"),
            value: SpaceValue::Dict(BTreeMap::from([
                ("a".to_string(), f32_tensor(&[0.5], vec![1])),
                ("b".to_string(), SpaceValue::Discrete(1)),
            ])),
            expect: Expect::Member,
        },
        Case {
            clause: "tuple with all children valid",
            space: TupleSpaceBuilder::new()
                .with(f32_box(0.0, 1.0, vec![2]))
                .with(DiscreteBuilder::new(3).build().expect("valid test fixture"))
                .build()
                .expect("valid test fixture"),
            value: SpaceValue::Tuple(vec![
                f32_tensor(&[0.0, 1.0], vec![2]),
                SpaceValue::Discrete(1),
            ]),
            expect: Expect::Member,
        },
        // ----- Structural deviations: always rejected -----
        Case {
            clause: "box wrong shape",
            space: f32_box(0.0, 1.0, vec![2]),
            value: f32_tensor(&[0.0], vec![1]),
            expect: Expect::Structural,
        },
        Case {
            clause: "box wrong dtype",
            space: f32_box(0.0, 1.0, vec![2]),
            value: SpaceValue::Box(
                Tensor::from_vec(vec![0u8; 16], vec![2], DType::Float64)
                    .expect("valid test fixture"),
            ),
            expect: Expect::Structural,
        },
        Case {
            clause: "box NaN within finite bounds",
            space: f32_box(0.0, 1.0, vec![2]),
            value: f32_tensor(&[f32::NAN, 0.5], vec![2]),
            expect: Expect::Structural,
        },
        Case {
            clause: "box NaN against unbounded space",
            space: f32_box(f64::NEG_INFINITY, f64::INFINITY, vec![2]),
            value: f32_tensor(&[f32::NAN, 0.5], vec![2]),
            expect: Expect::Structural,
        },
        Case {
            clause: "box NaN outranks an out-of-bounds element",
            space: f32_box(0.0, 1.0, vec![2]),
            value: f32_tensor(&[2.5, f32::NAN], vec![2]),
            expect: Expect::Structural,
        },
        Case {
            clause: "discrete above domain",
            space: DiscreteBuilder::new(4).build().expect("valid test fixture"),
            value: SpaceValue::Discrete(4),
            expect: Expect::Structural,
        },
        Case {
            clause: "discrete below start",
            space: DiscreteBuilder::new(4)
                .start(2)
                .build()
                .expect("valid test fixture"),
            value: SpaceValue::Discrete(1),
            expect: Expect::Structural,
        },
        Case {
            clause: "multidiscrete element outside its per-element domain",
            space: MultiDiscreteBuilder::vector([2, 3])
                .build()
                .expect("valid test fixture"),
            value: SpaceValue::MultiDiscrete(vec![1, 3]),
            expect: Expect::Structural,
        },
        Case {
            clause: "multidiscrete wrong arity",
            space: MultiDiscreteBuilder::vector([2, 3])
                .build()
                .expect("valid test fixture"),
            value: SpaceValue::MultiDiscrete(vec![1]),
            expect: Expect::Structural,
        },
        Case {
            clause: "multibinary wrong shape",
            space: MultiBinaryBuilder::scalar(3)
                .build()
                .expect("valid test fixture"),
            value: SpaceValue::MultiBinary(vec![true, false]),
            expect: Expect::Structural,
        },
        Case {
            clause: "dict missing a declared key",
            space: DictSpaceBuilder::new()
                .insert("a", unit_box())
                .insert(
                    "b",
                    DiscreteBuilder::new(3).build().expect("valid test fixture"),
                )
                .build()
                .expect("valid test fixture"),
            value: SpaceValue::Dict(BTreeMap::from([(
                "a".to_string(),
                f32_tensor(&[0.5], vec![1]),
            )])),
            expect: Expect::Structural,
        },
        Case {
            clause: "tuple wrong arity",
            space: TupleSpaceBuilder::new()
                .with(f32_box(0.0, 1.0, vec![2]))
                .with(DiscreteBuilder::new(3).build().expect("valid test fixture"))
                .build()
                .expect("valid test fixture"),
            value: SpaceValue::Tuple(vec![SpaceValue::Discrete(1)]),
            expect: Expect::Structural,
        },
        Case {
            // child "a" is out of bounds (range), child "b" has a NaN
            // (structural); structural outranks range across the whole value.
            clause: "dict structural child outranks a range sibling",
            space: DictSpaceBuilder::new()
                .insert("a", unit_box())
                .insert("b", unit_box())
                .build()
                .expect("valid test fixture"),
            value: SpaceValue::Dict(BTreeMap::from([
                ("a".to_string(), f32_tensor(&[2.5], vec![1])),
                ("b".to_string(), f32_tensor(&[f32::NAN], vec![1])),
            ])),
            expect: Expect::Structural,
        },
        // ----- Range deviations: policy-governed -----
        Case {
            clause: "box below low bound",
            space: f32_box(0.0, 1.0, vec![2]),
            value: f32_tensor(&[-0.5, 0.5], vec![2]),
            expect: Expect::Range,
        },
        Case {
            clause: "box above high bound",
            space: f32_box(0.0, 1.0, vec![2]),
            value: f32_tensor(&[0.5, 2.5], vec![2]),
            expect: Expect::Range,
        },
        Case {
            clause: "text below minimum length",
            space: TextBuilder::new(5)
                .min_length(2)
                .build()
                .expect("valid test fixture"),
            value: SpaceValue::Text("a".into()),
            expect: Expect::Range,
        },
        Case {
            clause: "text above maximum length",
            space: TextBuilder::new(3).build().expect("valid test fixture"),
            value: SpaceValue::Text("abcd".into()),
            expect: Expect::Range,
        },
        Case {
            clause: "text character outside charset",
            space: TextBuilder::new(5)
                .charset("abc")
                .build()
                .expect("valid test fixture"),
            value: SpaceValue::Text("abx".into()),
            expect: Expect::Range,
        },
        Case {
            clause: "dict propagates a child range deviation",
            space: DictSpaceBuilder::new()
                .insert("a", unit_box())
                .build()
                .expect("valid test fixture"),
            value: SpaceValue::Dict(BTreeMap::from([(
                "a".to_string(),
                f32_tensor(&[2.5], vec![1]),
            )])),
            expect: Expect::Range,
        },
        Case {
            clause: "tuple propagates a child range deviation",
            space: TupleSpaceBuilder::new()
                .with(f32_box(0.0, 1.0, vec![2]))
                .with(DiscreteBuilder::new(3).build().expect("valid test fixture"))
                .build()
                .expect("valid test fixture"),
            value: SpaceValue::Tuple(vec![
                f32_tensor(&[2.5, 0.5], vec![2]),
                SpaceValue::Discrete(1),
            ]),
            expect: Expect::Range,
        },
    ]
}

#[test]
fn conformance_contract_2026_06() {
    for case in cases() {
        let classified = match conform(&case.space, &case.value) {
            Conformance::Ok => Expect::Member,
            Conformance::Structural(_) => Expect::Structural,
            Conformance::Range(_) => Expect::Range,
        };
        assert_eq!(
            classified, case.expect,
            "clause `{}`: conform classified it as {classified:?}, expected {:?}",
            case.clause, case.expect,
        );

        // `contains` is the strict view (structural and range both required), so a
        // member is accepted and any deviation — structural or range — is rejected.
        let strict = contains(&case.space, &case.value);
        match case.expect {
            Expect::Member => assert!(
                strict.is_ok(),
                "clause `{}`: contains should accept a member, got {strict:?}",
                case.clause,
            ),
            Expect::Structural | Expect::Range => assert!(
                strict.is_err(),
                "clause `{}`: contains should reject a deviation",
                case.clause,
            ),
        }
    }
}

/// The serving side's range policy: `warn` (default) tolerates a range deviation
/// with a warning, `strict` rejects it, `off` accepts it silently — and every
/// policy still rejects structural deviations and accepts members.
#[test]
fn policy_application_2026_06() {
    use ValidationPolicy::{Off, Strict, Warn};

    let space = f32_box(0.0, 1.0, vec![2]);
    let member = f32_tensor(&[0.0, 1.0], vec![2]);
    let range = f32_tensor(&[0.5, 2.5], vec![2]);
    let structural = f32_tensor(&[f32::NAN, 0.5], vec![2]);

    // Members accepted and structural deviations rejected under every policy.
    for policy in [Warn, Strict, Off] {
        assert!(
            matches!(policy.check(&space, &member), PolicyOutcome::Accept),
            "{policy:?} should accept a member",
        );
        assert!(
            matches!(policy.check(&space, &structural), PolicyOutcome::Reject(_)),
            "{policy:?} must reject a structural deviation (policy-immune)",
        );
    }

    // Range deviations follow the policy.
    assert!(matches!(Warn.check(&space, &range), PolicyOutcome::Warn(_)));
    assert!(matches!(
        Strict.check(&space, &range),
        PolicyOutcome::Reject(_)
    ));
    assert!(matches!(Off.check(&space, &range), PolicyOutcome::Accept));

    // The default policy is Warn.
    assert!(matches!(
        ValidationPolicy::default().check(&space, &range),
        PolicyOutcome::Warn(_)
    ));
}
