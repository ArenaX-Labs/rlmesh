//! An image input expected by a model.

use std::collections::BTreeMap;
use std::fmt;

use serde::de::{self, SeqAccess, Visitor};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::spec::layouts::ImageLayout;
use crate::spec::{AcceptSet, FitMode};

fn default_uint8() -> String {
    "uint8".to_owned()
}

fn default_bilinear() -> String {
    "bilinear".to_owned()
}

/// Whether (and into what range) 8-bit pixels are mapped before the dtype cast.
///
/// One wire field with three forms, so the range can never disagree with an
/// on/off flag (the old `normalize` + `normalize_range` pair could):
/// `false` (the default) is off, `true` normalizes into the conventional
/// `[0, 1]`, and a `[min, max]` pair normalizes into that range. `false` is an
/// authoritative off-switch — there is no second field that can force it back on.
/// Mirrors `AcceptSet`'s scalar-or-list wire shape (here bool-or-pair).
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Normalize {
    /// No normalization (the default); pixels pass through to the dtype cast.
    #[default]
    Off,
    /// Normalize into the conventional `[0, 1]` range (wire: `true`).
    Unit,
    /// Normalize into an explicit `[min, max]` range (wire: `[min, max]`).
    Range(f64, f64),
}

impl Normalize {
    /// The `(min, max)` range to map `[0, 255]` into, or `None` when off.
    pub fn range(&self) -> Option<(f64, f64)> {
        match self {
            Normalize::Off => None,
            Normalize::Unit => Some((0.0, 1.0)),
            Normalize::Range(low, high) => Some((*low, *high)),
        }
    }
}

impl Serialize for Normalize {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Always emitted (like `resample`), so the resolved choice is explicit on
        // the wire and never diverges by reader default. `false`/`true` keep byte
        // parity with the old `normalize` bool; a range is a `[min, max]` pair.
        match self {
            Normalize::Off => serializer.serialize_bool(false),
            Normalize::Unit => serializer.serialize_bool(true),
            Normalize::Range(low, high) => {
                let mut seq = serializer.serialize_seq(Some(2))?;
                seq.serialize_element(low)?;
                seq.serialize_element(high)?;
                seq.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for Normalize {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct NormalizeVisitor;

        impl<'de> Visitor<'de> for NormalizeVisitor {
            type Value = Normalize;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a bool or a normalize range [min, max]")
            }

            fn visit_bool<E: de::Error>(self, value: bool) -> Result<Normalize, E> {
                Ok(if value {
                    Normalize::Unit
                } else {
                    Normalize::Off
                })
            }

            fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<Normalize, A::Error> {
                // Reuse the shared range deserializer so a `[min, max]` here gets
                // the same domain-friendly errors and reversed-range guard as
                // every other range field (see `spec::num::RangeVisitor`).
                let (low, high) = crate::spec::num::RangeVisitor.visit_seq(seq)?;
                Ok(Normalize::Range(low, high))
            }
        }

        deserializer.deserialize_any(NormalizeVisitor)
    }
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
    /// Whether (and into what range) 8-bit pixels are mapped before the dtype
    /// cast: `false` (off, the default), `true` (the conventional `[0, 1]`), or a
    /// `[min, max]` pair for a model trained on a different range (e.g.
    /// `[-1, 1]`). One field, so an on/off flag can never disagree with a range;
    /// `false` is an authoritative off-switch.
    #[serde(default)]
    pub normalize: Normalize,
    #[serde(default, deserialize_with = "crate::spec::num::de_count")]
    pub lead_dims: u32,
    #[serde(default)]
    pub upside_down: bool,
    /// Resize algorithm the model's training pipeline used. Defaults to
    /// `"bilinear"` (the plain half-pixel-center bilinear that torch/OpenCV
    /// pipelines use, which most trained policies match); set `"bilinear_aa"` for
    /// the antialiased PIL filter. A constrained string (not an enum) so future
    /// additive values degrade to a typed resolution error on older cores instead
    /// of a parse failure. Always emitted, so the resolved filter is explicit on
    /// the wire and never diverges by reader default.
    #[serde(default = "default_bilinear")]
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
    /// Unrecognized additive fields, retained for round-trip and surfaced to the
    /// publish-door `reject_unknowns` guard. See the strict-v1 publish gate.
    #[serde(flatten)]
    pub unknown: BTreeMap<String, serde_json::Value>,
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
    fn normalize_overloads_bool_and_range() {
        use crate::spec::model::image::Normalize;

        // Absent -> Off (the default), and Off serializes back as `false` (byte
        // parity with the old always-emitted `normalize` bool).
        let off = image("");
        let ModelLeaf::Image(img) = &off else {
            panic!("expected image")
        };
        assert_eq!(img.normalize, Normalize::Off);
        assert_eq!(img.normalize.range(), None);
        assert!(
            serde_json::to_string(&off)
                .unwrap()
                .contains("\"normalize\":false")
        );

        // `true` -> Unit -> [0, 1].
        let unit = image(r#", "normalize": true"#);
        let ModelLeaf::Image(img) = &unit else {
            panic!("expected image")
        };
        assert_eq!(img.normalize, Normalize::Unit);
        assert_eq!(img.normalize.range(), Some((0.0, 1.0)));
        assert!(
            serde_json::to_string(&unit)
                .unwrap()
                .contains("\"normalize\":true")
        );

        // A `[min, max]` pair -> Range, round-tripping as the pair.
        let signed = image(r#", "normalize": [-1.0, 1.0]"#);
        let ModelLeaf::Image(img) = &signed else {
            panic!("expected image")
        };
        assert_eq!(img.normalize, Normalize::Range(-1.0, 1.0));
        assert_eq!(img.normalize.range(), Some((-1.0, 1.0)));
        assert!(
            serde_json::to_string(&signed)
                .unwrap()
                .contains("\"normalize\":[-1.0,1.0]")
        );

        // A reversed range silently inverts pixel polarity; the shared range
        // deserializer rejects min > max at the wire boundary. A degenerate equal
        // range still parses.
        assert!(
            serde_json::from_str::<ModelLeaf>(
                r#"{"type": "image", "role": "image/primary", "normalize": [1.0, 0.0]}"#
            )
            .is_err()
        );
        assert!(
            serde_json::from_str::<ModelLeaf>(
                r#"{"type": "image", "role": "image/primary", "normalize": [0.5, 0.5]}"#
            )
            .is_ok()
        );
    }

    #[test]
    fn tagged_payload_captures_unknown_field_for_round_trip() {
        // Tolerant reader: a typo'd (or future-additive) field on a model leaf is
        // retained verbatim in `unknown`, not silently dropped. The known fields
        // still default; the publish-door `reject_unknowns` gate rejects the
        // stray field (see `spec::strict`).
        let input = image(r#", "layuot": "chw""#);
        let ModelLeaf::Image(img) = &input else {
            panic!("expected image")
        };
        assert_eq!(img.layout, ImageLayout::Hwc); // known field still defaults
        assert_eq!(img.unknown.get("layuot"), Some(&serde_json::json!("chw")));
        // Re-emitted verbatim, with `type` never leaking into the capture.
        assert!(!img.unknown.contains_key("type"));
        assert!(serde_json::to_string(&input).unwrap().contains("layuot"));
    }
}
