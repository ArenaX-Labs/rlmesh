//! A numeric state input expected by a model.

use serde::{Deserialize, Serialize};

use crate::spec::AcceptSet;
use crate::spec::rotations::RotationEncoding;

fn default_float32() -> String {
    "float32".to_owned()
}

/// One piece of a model state vector, sourced from an env state feature.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StateComponent {
    pub role: String,
    /// Rotation encoding(s) the model accepts for this piece, in preference
    /// order (most-preferred first). The resolver picks the env's native
    /// encoding when it appears here (no conversion), else converts the env's
    /// native into the first entry. A bare string on the wire for the common
    /// single-encoding case.
    #[serde(default)]
    pub encoding: Option<AcceptSet<RotationEncoding>>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_count")]
    pub dim: Option<u32>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_count")]
    pub index: Option<u32>,
    /// Target value range. When set and the env feature declares a (derived
    /// or tagged) source range, values are affinely mapped from the env
    /// range into this one — the state-side analogue of action range mapping.
    /// When the env feature has no source range (an unbounded/non-uniform space
    /// with no `range` tag) there is nothing to map from, so this is a no-op —
    /// it does not clamp or rescale on its own.
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_range")]
    pub range: Option<(f64, f64)>,
    #[serde(default)]
    pub optional: bool,
}

/// Container kind for a resolved state value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StateContainer {
    #[default]
    Array,
    List,
}

/// A numeric state input expected by a model.
///
/// Deserialization goes through `StateInputWire` so an empty `components` list
/// is rejected by the authoritative codec — matching Python
/// `StateInput.__post_init__` (which needs `components=(...)` or a single
/// `role=`). Without this the normalize/publish door would bless a state input
/// the read path crashes on. Mirrors the env-side `StateLayoutWire` empty guard.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "StateInputWire")]
pub struct StateInput {
    pub key: String,
    pub components: Vec<StateComponent>,
    #[serde(default)]
    pub pad_to: Option<u32>,
    #[serde(default = "default_float32")]
    pub dtype: String,
    #[serde(default)]
    pub reshape: Option<Vec<i64>>,
    #[serde(default)]
    pub container: StateContainer,
}

/// Wire form of [`StateInput`]; see its docs for the non-empty-components rule.
#[derive(Deserialize)]
struct StateInputWire {
    key: String,
    components: Vec<StateComponent>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_count")]
    pad_to: Option<u32>,
    #[serde(default = "default_float32")]
    dtype: String,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_dims")]
    reshape: Option<Vec<i64>>,
    #[serde(default)]
    container: StateContainer,
}

impl TryFrom<StateInputWire> for StateInput {
    type Error = String;

    fn try_from(wire: StateInputWire) -> Result<Self, Self::Error> {
        if wire.components.is_empty() {
            return Err("a state input needs at least one component".to_owned());
        }
        for component in &wire.components {
            // `index` selects one element and `dim` truncates to the leading N;
            // apply applies `index` and silently ignores `dim` when both are set,
            // so reject the ambiguous pairing at the codec instead of picking one.
            if component.dim.is_some() && component.index.is_some() {
                return Err(format!(
                    "state component {:?} sets both dim and index; index selects one element \
                     and dim truncates to the leading N -- set one, not both",
                    component.role
                ));
            }
        }
        Ok(StateInput {
            key: wire.key,
            components: wire.components,
            pad_to: wire.pad_to,
            dtype: wire.dtype,
            reshape: wire.reshape,
            container: wire.container,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::StateInput;

    #[test]
    fn rejects_empty_components() {
        // Parity with Python StateInput.__post_init__: a state input with no
        // components is rejected at the codec, so the publish door never blesses
        // a spec the read path cannot reconstruct.
        let err =
            serde_json::from_str::<StateInput>(r#"{"key": "s", "components": []}"#).unwrap_err();
        assert!(
            err.to_string().contains("at least one component"),
            "got: {err}"
        );
        let ok: StateInput = serde_json::from_str(r#"{"key": "s", "components": [{"role": "r"}]}"#)
            .expect("non-empty parses");
        assert_eq!(ok.components.len(), 1);
    }

    #[test]
    fn rejects_component_with_both_dim_and_index() {
        let err = serde_json::from_str::<StateInput>(
            r#"{"key": "s", "components": [{"role": "r", "dim": 3, "index": 0}]}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("both dim and index"), "got: {err}");
    }

    #[test]
    fn rejects_invalid_reshape_dims() {
        // reshape elements are a concrete size or a single -1 (infer); the codec
        // rejects what apply would error on per-step, keeping the publish door
        // from blessing a spec the read path crashes on.
        let err = serde_json::from_str::<StateInput>(
            r#"{"key": "s", "components": [{"role": "r"}], "reshape": [-5]}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("infer"), "got: {err}");
        let err = serde_json::from_str::<StateInput>(
            r#"{"key": "s", "components": [{"role": "r"}], "reshape": [-1, -1]}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("at most one -1"), "got: {err}");
        let ok: StateInput = serde_json::from_str(
            r#"{"key": "s", "components": [{"role": "r"}], "reshape": [-1, 4]}"#,
        )
        .expect("one infer parses");
        assert_eq!(ok.reshape, Some(vec![-1, 4]));
    }
}
