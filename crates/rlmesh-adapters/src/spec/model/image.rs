//! An image input expected by a model.

use serde::{Deserialize, Serialize};

use crate::spec::layouts::ImageLayout;
use crate::spec::{AcceptSet, FitMode};

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

/// Deserialize `stack`, enforcing the `1..=MAX_STACK` bound at the wire boundary
/// (via the shared bounded-count helper; see
/// [`de_bounded_count`](crate::spec::num::de_bounded_count)).
fn de_stack<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    crate::spec::num::de_bounded_count(deserializer, "stack", MAX_STACK)
}

/// An image input expected by a model.
///
/// There is no `key` — placement is the tree position this leaf sits at.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Image {
    pub role: String,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_count")]
    pub height: Option<u32>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_count")]
    pub width: Option<u32>,
    #[serde(default)]
    pub layout: ImageLayout,
    /// Channel count the model expects (e.g. `3` for RGB, `1` for grayscale).
    /// When set, a resolve error if the env image has a different channel count
    /// — the adapter does not (yet) convert between channel counts, so this
    /// turns a silent wrong-channel feed into a loud failure. Additive over the
    /// pinned wire format (omitted when unset).
    #[serde(
        default,
        deserialize_with = "crate::spec::num::de_opt_count",
        skip_serializing_if = "Option::is_none"
    )]
    pub channels: Option<u32>,
    #[serde(default = "default_uint8")]
    pub dtype: String,
    /// Map 8-bit pixels into `normalize_range` (default `[0, 1]`) before casting.
    /// Setting `normalize_range` alone also turns normalization on, so this flag
    /// is only needed to request the default `[0, 1]` range without spelling it
    /// out (it is *not* an off-switch for a declared range).
    #[serde(default)]
    pub normalize: bool,
    /// Target value range, mapped from `[0, 255]`. Setting it implies
    /// normalization even when `normalize` is false. Defaults to `[0, 1]` (the
    /// conventional 8-bit normalization); set e.g. `[-1, 1]` for a model trained
    /// on signed inputs. Additive over the pinned wire format (omitted when unset).
    #[serde(
        default,
        deserialize_with = "crate::spec::num::de_opt_range",
        skip_serializing_if = "Option::is_none"
    )]
    pub normalize_range: Option<(f64, f64)>,
    #[serde(default, deserialize_with = "crate::spec::num::de_count")]
    pub lead_dims: u32,
    #[serde(default)]
    pub upside_down: bool,
    /// Resize algorithm the model's training pipeline used. A constrained
    /// string (not an enum) so future additive values degrade to a typed
    /// resolution error on older cores instead of a parse failure.
    #[serde(default = "default_bilinear_aa")]
    pub resample: String,
    /// Permit the resize to *upscale* (interpolate detail the env image does not
    /// have). Off by default: a model target larger than the env's native
    /// resolution is a resolve error unless this is set. Additive over the pinned
    /// wire format (omitted when false).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub allow_upscale: bool,
    /// How to reconcile a target whose aspect ratio differs from the env image.
    /// A single mode (`"stretch"`, `"crop"`, or `"pad"`) or a preference list
    /// (`["crop", "pad"]`): the resolver picks, per env, the first that does not
    /// need a disallowed upscale — so one spec can crop a large camera and
    /// letterbox a small one. Required only when the aspects differ; absent it,
    /// an aspect-changing resize is a resolve error (no silent distortion). An
    /// unrecognized mode degrades (it is skipped), never a parse failure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit: Option<AcceptSet<FitMode>>,
    /// Zero-fill a black frame when the env does not provide this camera, instead
    /// of failing resolution. Needs `height`, `width`, and `channels` so the
    /// blank can be sized without an env image. Additive over the pinned wire
    /// format (omitted when false). Mirrors a [`ConcatPart`]'s `optional`.
    ///
    /// [`ConcatPart`]: crate::spec::ConcatPart
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub optional: bool,
    /// Raw 8-bit fill level for an absent `optional` camera: `0` = black (the
    /// default), `255` = white, `128` ~ mid-gray. Applied before the
    /// normalize/dtype steps, so it lands wherever that level maps in the model's
    /// range. Additive over the pinned wire format (omitted when unset).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub absent_fill: Option<u8>,
    /// Number of consecutive observations the model stacks on a new leading
    /// axis (frame history); `1` = no stacking. Stacking is applied natively in
    /// the core, episode-keyed (the env still sends one frame per step; the
    /// per-episode rolling window lives in `rlmesh_adapters::stateful`). Omitted
    /// from the wire when `1` to stay byte-identical with the Python serializer;
    /// bounded to `MAX_STACK`.
    #[serde(
        default = "crate::spec::num::default_one",
        deserialize_with = "de_stack",
        skip_serializing_if = "crate::spec::num::is_one"
    )]
    pub stack: u32,
}

#[cfg(test)]
mod tests {
    use crate::spec::{ImageLayout, ModelLeaf};

    fn image(extra: &str) -> ModelLeaf {
        let json = format!(r#"{{"type": "image", "role": "image/primary"{extra}}}"#);
        serde_json::from_str(&json).expect("parse")
    }

    #[test]
    fn stack_defaults_to_one_and_is_omitted_from_wire() {
        let input = image("");
        let ModelLeaf::Image(img) = &input else {
            panic!("expected image")
        };
        assert_eq!(img.stack, 1);
        // Byte parity with the Python serializer: stack omitted when 1.
        assert!(!serde_json::to_string(&input).unwrap().contains("stack"));
    }

    #[test]
    fn stack_roundtrips_when_set() {
        let input = image(r#", "stack": 4"#);
        let ModelLeaf::Image(img) = &input else {
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
            serde_json::from_str::<ModelLeaf>(
                r#"{"type": "image", "role": "image/primary", "stack": 0}"#
            )
            .is_err()
        );
        assert!(
            serde_json::from_str::<ModelLeaf>(
                r#"{"type": "image", "role": "image/primary", "stack": 1000}"#
            )
            .is_err()
        );
    }

    #[test]
    fn absent_fill_defaults_to_none_and_is_omitted_from_wire() {
        let input = image("");
        let ModelLeaf::Image(img) = &input else {
            panic!("expected image")
        };
        assert_eq!(img.absent_fill, None);
        assert!(
            !serde_json::to_string(&input)
                .unwrap()
                .contains("absent_fill")
        );
    }

    #[test]
    fn absent_fill_roundtrips_when_set() {
        let input = image(r#", "absent_fill": 128"#);
        let ModelLeaf::Image(img) = &input else {
            panic!("expected image")
        };
        assert_eq!(img.absent_fill, Some(128));
        assert!(
            serde_json::to_string(&input)
                .unwrap()
                .contains("\"absent_fill\":128")
        );
    }

    #[test]
    fn absent_fill_rejects_out_of_range() {
        // A `u8` gives free 0..=255 validation at the codec door.
        assert!(
            serde_json::from_str::<ModelLeaf>(
                r#"{"type": "image", "role": "image/primary", "absent_fill": 300}"#
            )
            .is_err()
        );
    }

    #[test]
    fn normalize_range_rejects_reversed_and_accepts_valid() {
        // A reversed range silently inverts pixel polarity at serve time; the
        // shared range deserializer rejects min > max at the wire boundary.
        assert!(
            serde_json::from_str::<ModelLeaf>(
                r#"{"type": "image", "role": "image/primary", "normalize_range": [1.0, 0.0]}"#
            )
            .is_err()
        );
        // A normal (and a degenerate equal) range still parse.
        let signed = image(r#", "normalize_range": [-1.0, 1.0]"#);
        let ModelLeaf::Image(img) = &signed else {
            panic!("expected image")
        };
        assert_eq!(img.normalize_range, Some((-1.0, 1.0)));
        assert!(
            serde_json::from_str::<ModelLeaf>(
                r#"{"type": "image", "role": "image/primary", "normalize_range": [0.5, 0.5]}"#
            )
            .is_ok()
        );
    }

    #[test]
    fn tagged_payload_does_not_reject_unknown_field_yet() {
        // A typo'd field on a ModelLeaf variant is still silently dropped: the
        // variant structs do not (yet) carry deny_unknown_fields, unlike the
        // model-side plain structs (Actuator, ConcatPart) that do.
        let input = image(r#", "layuot": "chw""#);
        let ModelLeaf::Image(img) = &input else {
            panic!("expected image")
        };
        assert_eq!(img.layout, ImageLayout::Hwc); // typo'd "layuot" ignored
    }
}
