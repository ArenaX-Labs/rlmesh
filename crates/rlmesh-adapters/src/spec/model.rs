//! The model-side spec: expected input payload plus the action output.

mod custom;
mod image;
mod state;
mod text;

use serde::{Deserialize, Serialize};

use super::action::ActionLayout;

pub use custom::CustomInput;
pub use image::ImageInput;
pub use state::{StateComponent, StateContainer, StateInput};
pub use text::{TextContainer, TextInput};

/// One input feature expected by a model, declared by the model.
///
/// **Strict v1 kind tag.** A new input *kind* (a new variant here) is a
/// structural change = a v2 key bump, not an additive v1 value; an unknown
/// `type` is rejected at parse by design (the value-vocabulary degradation that
/// applies to [`crate::spec::RotationEncoding`] is deliberately NOT extended to
/// node kinds — a new kind has no defined structure for an old reader).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ModelInput {
    Image(ImageInput),
    State(StateInput),
    Text(TextInput),
    Custom(CustomInput),
}

/// Declarative description of a model's input payload and action output.
///
/// Deserialization goes through `ModelSpecWire` so duplicate input keys are
/// rejected by the authoritative codec — matching Python
/// `ModelSpec.__post_init__` and Rust `resolve` (both of which reject a repeated
/// key). Without this the normalize/publish door would bless a spec the read
/// path and resolve cannot consume.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "ModelSpecWire")]
pub struct ModelSpec {
    pub inputs: Vec<ModelInput>,
    pub action: ActionLayout,
}

/// Wire form of [`ModelSpec`]; see its docs for the duplicate-key rule.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelSpecWire {
    inputs: Vec<ModelInput>,
    action: ActionLayout,
}

impl TryFrom<ModelSpecWire> for ModelSpec {
    type Error = String;

    fn try_from(wire: ModelSpecWire) -> Result<Self, Self::Error> {
        let mut seen = std::collections::BTreeSet::new();
        for input in &wire.inputs {
            let key = match input {
                ModelInput::Image(input) => &input.key,
                ModelInput::State(input) => &input.key,
                ModelInput::Text(input) => &input.key,
                ModelInput::Custom(input) => &input.key,
            };
            if !seen.insert(key.as_str()) {
                return Err(format!("duplicate model input key {key:?}"));
            }
        }
        Ok(ModelSpec {
            inputs: wire.inputs,
            action: wire.action,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::ModelSpec;

    #[test]
    fn rejects_duplicate_input_key() {
        // Parity with Python ModelSpec.__post_init__ and Rust resolve: two
        // inputs sharing a key are rejected at the codec, so the publish door
        // never blesses a spec the read path / resolve cannot consume.
        let err = serde_json::from_str::<ModelSpec>(
            r#"{"inputs":[{"type":"text","key":"s","role":"r"},{"type":"text","key":"s","role":"r"}],"action":{"components":[]}}"#,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("duplicate model input key"),
            "got: {err}"
        );
        let ok: ModelSpec = serde_json::from_str(
            r#"{"inputs":[{"type":"text","key":"a","role":"r"},{"type":"text","key":"b","role":"r"}],"action":{"components":[]}}"#,
        )
        .expect("distinct keys parse");
        assert_eq!(ok.inputs.len(), 2);
    }
}
