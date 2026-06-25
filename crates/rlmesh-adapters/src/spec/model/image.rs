//! An image input expected by a model.

use serde::{Deserialize, Serialize};

use crate::spec::layouts::ImageLayout;

fn default_uint8() -> String {
    "uint8".to_owned()
}

fn default_bilinear_aa() -> String {
    "bilinear_aa".to_owned()
}

/// Upper bound on frame-stacking depth (mirrors the Python `_MAX_STACK`). A
/// spec can arrive from an untrusted contract; without a ceiling a huge `stack`
/// would make the host adapter buffer that many frames and exhaust memory.
const MAX_STACK: u32 = 64;

fn default_stack() -> u32 {
    1
}

fn stack_is_default(stack: &u32) -> bool {
    *stack == 1
}

/// Deserialize `stack`, enforcing the `1..=MAX_STACK` bound at the wire
/// boundary. Routes through [`de_count`](crate::spec::num::de_count) so a
/// negative/non-integer reads in domain language too (no leaked `u32`).
fn de_stack<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let stack = crate::spec::num::de_count(deserializer)?;
    if !(1..=MAX_STACK).contains(&stack) {
        return Err(serde::de::Error::custom(format!(
            "stack must be between 1 and {MAX_STACK}, got {stack}"
        )));
    }
    Ok(stack)
}

/// An image input expected by a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageInput {
    pub key: String,
    pub role: String,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_count")]
    pub height: Option<u32>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_count")]
    pub width: Option<u32>,
    #[serde(default)]
    pub layout: ImageLayout,
    #[serde(default = "default_uint8")]
    pub dtype: String,
    #[serde(default)]
    pub normalize: bool,
    #[serde(default, deserialize_with = "crate::spec::num::de_count")]
    pub lead_dims: u32,
    #[serde(default)]
    pub upside_down: bool,
    /// Resize algorithm the model's training pipeline used. A constrained
    /// string (not an enum) so future additive values degrade to a typed
    /// resolution error on older cores instead of a parse failure.
    #[serde(default = "default_bilinear_aa")]
    pub resample: String,
    /// Number of consecutive observations the model stacks on a new leading
    /// axis (frame history); `1` = no stacking. Stacking is applied natively in
    /// the core, episode-keyed (the env still sends one frame per step; the
    /// per-episode rolling window lives in `rlmesh_adapters::stateful`). Omitted
    /// from the wire when `1` to stay byte-identical with the Python serializer;
    /// bounded to `MAX_STACK`.
    #[serde(
        default = "default_stack",
        deserialize_with = "de_stack",
        skip_serializing_if = "stack_is_default"
    )]
    pub stack: u32,
}

#[cfg(test)]
mod tests {
    use crate::spec::{ImageLayout, ModelInput};

    fn image(extra: &str) -> ModelInput {
        let json = format!(r#"{{"type": "image", "key": "cam", "role": "image/primary"{extra}}}"#);
        serde_json::from_str(&json).expect("parse")
    }

    #[test]
    fn stack_defaults_to_one_and_is_omitted_from_wire() {
        let input = image("");
        let ModelInput::Image(img) = &input else {
            panic!("expected image")
        };
        assert_eq!(img.stack, 1);
        // Byte parity with the Python serializer: stack omitted when 1.
        assert!(!serde_json::to_string(&input).unwrap().contains("stack"));
    }

    #[test]
    fn stack_roundtrips_when_set() {
        let input = image(r#", "stack": 4"#);
        let ModelInput::Image(img) = &input else {
            panic!("expected image")
        };
        assert_eq!(img.stack, 4);
        assert!(
            serde_json::to_string(&input)
                .unwrap()
                .contains("\"stack\":4")
        );
    }

    #[test]
    fn stack_bound_enforced() {
        assert!(
            serde_json::from_str::<ModelInput>(
                r#"{"type": "image", "key": "cam", "role": "image/primary", "stack": 0}"#
            )
            .is_err()
        );
        assert!(
            serde_json::from_str::<ModelInput>(
                r#"{"type": "image", "key": "cam", "role": "image/primary", "stack": 1000}"#
            )
            .is_err()
        );
    }

    #[test]
    fn tagged_payload_does_not_reject_unknown_field_yet() {
        // Documents the serde limitation: deny_unknown_fields cannot apply to an
        // internally-tagged variant payload, so a typo'd field here is silently
        // dropped until the Rust normalize door adds a manual key check. If this
        // ever starts rejecting, the limitation was lifted -- update the policy.
        let input = image(r#", "layuot": "chw""#);
        let ModelInput::Image(img) = &input else {
            panic!("expected image")
        };
        assert_eq!(img.layout, ImageLayout::Hwc); // typo'd "layuot" ignored
    }
}
