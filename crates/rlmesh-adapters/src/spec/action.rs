//! Action layout types shared by env and model declarations.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::rotations::RotationEncoding;

/// One contiguous slice of an action vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Actuator {
    /// A role-less actuator is *opaque*: it occupies `dim` dims of the action
    /// vector with the constant `fill`, matched by no model role -- the
    /// action-side mirror of a role-less `Field`. An absent role on the wire is
    /// the opaque case; a present role is the normal model-mapped case.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    // `dim` is required (no `default`): an absent `dim` is a missing-field error
    // at the codec boundary, not a silent 0 that surfaces later as a confusing
    // width-sum mismatch. Mirrors StateFieldWire, which also omits `default`.
    #[serde(deserialize_with = "crate::spec::num::de_count")]
    pub dim: u32,
    #[serde(default)]
    pub encoding: Option<RotationEncoding>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_range")]
    pub range: Option<(f64, f64)>,
    #[serde(default)]
    pub binary: bool,
    // scale/invert/threshold are additive env-side corrections; they are
    // omitted from serialization when unset so layouts that do not use them are
    // byte-identical to before (matching the Python serializer). scale/threshold
    // route through de_opt_number so a wrong-typed value reads in domain language
    // (`expected a number`) instead of leaking the Rust wire type `f64`.
    // `invert` is sugar: it negates, so it is exactly `scale = -scale` (a lone
    // `invert` == `scale = -1`). Kept as an explicit gripper-sign knob; do not
    // add further sign knobs.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::spec::num::de_opt_number"
    )]
    pub scale: Option<f64>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub invert: bool,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "crate::spec::num::de_opt_number"
    )]
    pub threshold: Option<f64>,
    /// Env-side per-actuator safety clamp: clamp this component's mapped value to
    /// its declared `range` (the global `Action.clip` cannot, since it applies one
    /// bound to a whole mixed-range vector). Resolve enforces that `clip` implies
    /// `range`. Omitted when unset for byte-parity with layouts that do not use it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub clip: bool,
    /// Constant emitted for each dim of a role-less (opaque) actuator -- the env
    /// requires these dims but no model produces them. Inert on a roled actuator
    /// (must stay 0.0; `reject` enforces this). Omitted when 0.0.
    #[serde(default, skip_serializing_if = "is_default_fill")]
    pub fill: f64,
    /// Unrecognized additive fields, retained for round-trip and surfaced to the
    /// publish-door `reject_unknowns` guard. See the strict-v1 publish gate.
    #[serde(flatten)]
    pub unknown: BTreeMap<String, serde_json::Value>,
}

/// A fill of 0.0 is the default (a roled actuator's inert value), omitted on the
/// wire so layouts that do not use an opaque actuator are byte-identical.
fn is_default_fill(fill: &f64) -> bool {
    *fill == 0.0
}

/// Ordered action components plus optional clipping bounds.
///
/// Deserialization goes through `ActionWire` so a duplicate component
/// role is rejected by the authoritative codec — matching Rust `resolve`
/// (`plan_action`), which rejects a repeated action role (it would build the
/// action by repetition instead of a real mapping). Without this the
/// normalize/publish door would bless a layout resolve cannot consume.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "ActionWire")]
pub struct Action {
    pub components: Vec<Actuator>,
    #[serde(default)]
    pub clip: Option<(f64, f64)>,
}

/// Wire form of [`Action`]; see its docs for the duplicate-role rule.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ActionWire {
    components: Vec<Actuator>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_range")]
    clip: Option<(f64, f64)>,
}

impl TryFrom<ActionWire> for Action {
    type Error = String;

    fn try_from(wire: ActionWire) -> Result<Self, Self::Error> {
        let mut seen = std::collections::BTreeSet::new();
        for component in &wire.components {
            if !component.fill.is_finite() {
                return Err(format!("actuator {:?} fill must be finite", component.role));
            }
            match &component.role {
                Some(role) => {
                    if !seen.insert(role.as_str()) {
                        return Err(format!(
                            "an action layout declares role {role:?} more than once"
                        ));
                    }
                    if component.fill != 0.0 {
                        return Err(format!(
                            "actuator {role:?}: fill applies only to a role-less (opaque) \
                             actuator; a roled actuator takes its values from the model"
                        ));
                    }
                }
                // A role-less (opaque) actuator emits a constant, so the
                // model-mapping fields are meaningless -- it carries only dim and
                // fill (the action-side mirror of a role-less Field's skip rule).
                None => {
                    if component.encoding.is_some()
                        || component.range.is_some()
                        || component.scale.is_some()
                        || component.invert
                        || component.threshold.is_some()
                        || component.binary
                        || component.clip
                    {
                        return Err("a role-less (opaque) actuator carries only dim and \
                             fill; drop encoding/range/scale/invert/threshold/binary/clip"
                            .to_owned());
                    }
                }
            }
        }
        Ok(Action {
            components: wire.components,
            clip: wire.clip,
        })
    }
}

#[cfg(test)]
mod finiteness_contract {
    use super::Action;

    // The v1 wire contract: range/clip/scale/threshold bounds are finite; an
    // unbounded bound is omitted, never +/-Infinity. serde_json enforces this
    // on the *consume* side for free -- it rejects the non-RFC-8259
    // `Infinity`/`NaN` tokens AND float literals that overflow to infinity
    // (`1e400`), and `serde_json::Number` cannot even hold a non-finite value.
    // These tests PIN that behavior so a future serde_json that started
    // silently importing an infinity fails loudly here. The *produce* side is
    // guarded in Python (finiteness checks at spec construction + allow_nan=False).

    #[test]
    fn rejects_infinity_token() {
        assert!(
            serde_json::from_str::<Action>(r#"{"components": [], "clip": [Infinity, 1.0]}"#)
                .is_err()
        );
    }

    #[test]
    fn rejects_overflow_literal() {
        assert!(
            serde_json::from_str::<Action>(r#"{"components": [], "clip": [1e400, 1.0]}"#).is_err()
        );
        assert!(
            serde_json::from_str::<Action>(
                r#"{"components": [{"role": "g", "dim": 1, "scale": 1e400}]}"#
            )
            .is_err()
        );
    }

    #[test]
    fn accepts_finite_bounds() {
        let ok: Action =
            serde_json::from_str(r#"{"components": [], "clip": [-1.0, 1.0]}"#).unwrap();
        assert_eq!(ok.clip, Some((-1.0, 1.0)));
    }
}

#[cfg(test)]
mod tolerant_field_contract {
    use super::{Action, Actuator};

    #[test]
    fn actuator_captures_unknown_field_for_round_trip() {
        // Tolerant reader: a typo'd (or future-additive) Actuator field is no
        // longer rejected at the serde layer — it is captured verbatim in
        // `unknown` and re-emitted, so a newer peer's field survives an older
        // core. The strict-v1 gate (`reject_unknowns`) catches it at the publish
        // door instead; see `spec::strict`.
        let actuator: Actuator =
            serde_json::from_str(r#"{"role": "x", "dim": 1, "rnge": [0.0, 1.0]}"#).unwrap();
        assert_eq!(
            actuator.unknown.get("rnge"),
            Some(&serde_json::json!([0.0, 1.0]))
        );
        // Round-trips verbatim.
        let json = serde_json::to_string(&actuator).unwrap();
        assert!(json.contains("rnge"), "unknown field dropped: {json}");
    }

    #[test]
    fn action_envelope_stays_strict() {
        // The Action/ActionWire envelope keeps deny_unknown_fields (it has a
        // cross-field TryFrom validator and is not a growable leaf), so a typo on
        // the envelope is still a hard parse error.
        let err =
            serde_json::from_str::<Action>(r#"{"components": [], "clipp": null}"#).unwrap_err();
        assert!(err.to_string().contains("unknown field"), "got: {err}");
    }
}

#[cfg(test)]
mod opaque_actuator_contract {
    use super::Action;

    #[test]
    fn role_less_actuator_carries_only_dim_and_fill() {
        // A role-less (opaque) actuator: only dim + fill. Absent role => opaque.
        let ok: Action =
            serde_json::from_str(r#"{"components": [{"dim": 2, "fill": 0.25}]}"#).unwrap();
        assert_eq!(ok.components[0].role, None);
        assert_eq!(ok.components[0].fill, 0.25);

        // A model-mapping field on a role-less actuator is rejected.
        let err =
            serde_json::from_str::<Action>(r#"{"components": [{"dim": 2, "range": [-1.0, 1.0]}]}"#)
                .unwrap_err();
        assert!(err.to_string().contains("role-less"), "{err}");
    }

    #[test]
    fn fill_on_a_roled_actuator_is_rejected() {
        // fill is the opaque constant; a roled actuator takes its values from the
        // model, so a non-zero fill there is a contradiction.
        let err = serde_json::from_str::<Action>(
            r#"{"components": [{"role": "g", "dim": 1, "fill": 0.5}]}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("fill applies only"), "{err}");
    }

    #[test]
    fn role_less_actuators_do_not_collide_on_the_dup_check() {
        // Two opaque actuators are fine (no role to repeat), mirroring role-less
        // Fields in a Split.
        let ok: Action =
            serde_json::from_str(r#"{"components": [{"dim": 1}, {"dim": 2}]}"#).unwrap();
        assert_eq!(ok.components.len(), 2);
    }
}

#[cfg(test)]
mod dup_role_contract {
    use super::Action;

    #[test]
    fn rejects_duplicate_component_role() {
        // Parity with Rust resolve (plan_action): two components sharing a role
        // are rejected at the codec, so the publish door never blesses a layout
        // resolve cannot consume. An empty layout stays valid (no action mapping).
        let err = serde_json::from_str::<Action>(
            r#"{"components": [{"role": "g", "dim": 1}, {"role": "g", "dim": 1}]}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("more than once"), "got: {err}");

        let ok: Action = serde_json::from_str(r#"{"components": []}"#).expect("empty parses");
        assert!(ok.components.is_empty());
        let ok: Action = serde_json::from_str(
            r#"{"components": [{"role": "a", "dim": 1}, {"role": "b", "dim": 1}]}"#,
        )
        .expect("distinct roles parse");
        assert_eq!(ok.components.len(), 2);
    }
}
