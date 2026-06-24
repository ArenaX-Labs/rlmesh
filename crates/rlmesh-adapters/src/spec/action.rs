//! Action layout types shared by env and model declarations.

use serde::{Deserialize, Serialize};

use super::rotations::RotationEncoding;

/// One contiguous slice of an action vector.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionComponent {
    pub role: String,
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
}

/// Ordered action components plus optional clipping bounds.
///
/// Deserialization goes through `ActionLayoutWire` so a duplicate component
/// role is rejected by the authoritative codec — matching Rust `resolve`
/// (`plan_action`), which rejects a repeated action role (it would build the
/// action by repetition instead of a real mapping). Without this the
/// normalize/publish door would bless a layout resolve cannot consume.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "ActionLayoutWire")]
pub struct ActionLayout {
    pub components: Vec<ActionComponent>,
    #[serde(default)]
    pub clip: Option<(f64, f64)>,
}

/// Wire form of [`ActionLayout`]; see its docs for the duplicate-role rule.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ActionLayoutWire {
    components: Vec<ActionComponent>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_range")]
    clip: Option<(f64, f64)>,
}

impl TryFrom<ActionLayoutWire> for ActionLayout {
    type Error = String;

    fn try_from(wire: ActionLayoutWire) -> Result<Self, Self::Error> {
        let mut seen = std::collections::BTreeSet::new();
        for component in &wire.components {
            if !seen.insert(component.role.as_str()) {
                return Err(format!(
                    "an action layout declares role {:?} more than once",
                    component.role
                ));
            }
        }
        Ok(ActionLayout {
            components: wire.components,
            clip: wire.clip,
        })
    }
}

#[cfg(test)]
mod finiteness_contract {
    use super::ActionLayout;

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
            serde_json::from_str::<ActionLayout>(r#"{"components": [], "clip": [Infinity, 1.0]}"#)
                .is_err()
        );
    }

    #[test]
    fn rejects_overflow_literal() {
        assert!(
            serde_json::from_str::<ActionLayout>(r#"{"components": [], "clip": [1e400, 1.0]}"#)
                .is_err()
        );
        assert!(
            serde_json::from_str::<ActionLayout>(
                r#"{"components": [{"role": "g", "dim": 1, "scale": 1e400}]}"#
            )
            .is_err()
        );
    }

    #[test]
    fn accepts_finite_bounds() {
        let ok: ActionLayout =
            serde_json::from_str(r#"{"components": [], "clip": [-1.0, 1.0]}"#).unwrap();
        assert_eq!(ok.clip, Some((-1.0, 1.0)));
    }
}

#[cfg(test)]
mod deny_unknown_fields_contract {
    use super::{ActionComponent, ActionLayout};

    #[test]
    fn plain_struct_rejects_typo() {
        // ActionComponent/ActionLayout are plain structs -> deny_unknown_fields
        // turns a field typo into a hard error instead of a silent drop.
        let err = serde_json::from_str::<ActionComponent>(
            r#"{"role": "x", "dim": 1, "rnge": [0.0, 1.0]}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("unknown field"), "got: {err}");

        let err = serde_json::from_str::<ActionLayout>(r#"{"components": [], "clipp": null}"#)
            .unwrap_err();
        assert!(err.to_string().contains("unknown field"), "got: {err}");
    }
}

#[cfg(test)]
mod dup_role_contract {
    use super::ActionLayout;

    #[test]
    fn rejects_duplicate_component_role() {
        // Parity with Rust resolve (plan_action): two components sharing a role
        // are rejected at the codec, so the publish door never blesses a layout
        // resolve cannot consume. An empty layout stays valid (no action mapping).
        let err = serde_json::from_str::<ActionLayout>(
            r#"{"components": [{"role": "g", "dim": 1}, {"role": "g", "dim": 1}]}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("more than once"), "got: {err}");

        let ok: ActionLayout = serde_json::from_str(r#"{"components": []}"#).expect("empty parses");
        assert!(ok.components.is_empty());
        let ok: ActionLayout = serde_json::from_str(
            r#"{"components": [{"role": "a", "dim": 1}, {"role": "b", "dim": 1}]}"#,
        )
        .expect("distinct roles parse");
        assert_eq!(ok.components.len(), 2);
    }
}
