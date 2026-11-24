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

/// How the `crop` / `crop_area` box is taken: `"zoom"` resamples the box
/// straight to the target (the box *is* the frame), `"slice"` cuts the box out
/// at integer pixels first and resizes that.
pub const CROP_MODES: [&str; 2] = ["zoom", "slice"];

/// The channel orders a model can be fed in.
pub const CHANNEL_ORDERS: [&str; 2] = ["rgb", "bgr"];

fn default_zoom() -> String {
    "zoom".to_owned()
}

fn default_rgb() -> String {
    "rgb".to_owned()
}

fn is_zoom(mode: &str) -> bool {
    mode == "zoom"
}

fn is_rgb(order: &str) -> bool {
    order == "rgb"
}

/// Whether (and into what range) 8-bit pixels are mapped before the dtype cast.
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
                let (low, high) = crate::spec::num::RangeVisitor.visit_seq(seq)?;
                Ok(Normalize::Range(low, high))
            }
        }

        deserializer.deserialize_any(NormalizeVisitor)
    }
}

pub(crate) const MAX_STACK: u32 = 64;

/// How the start of an episode fills a frame window that has not seen enough
/// steps to be full.
///
/// A `Black` pad frame goes through the plan like [`Image::fill`]: under
/// `normalize = [-1, 1]` it is `-1.0`, not `0.0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StackPad {
    /// Replicate the first observed frame of the episode.
    #[default]
    First,
    /// A black (raw 8-bit `0`) frame pushed through the plan.
    Black,
}

impl StackPad {
    fn is_first(&self) -> bool {
        matches!(self, StackPad::First)
    }
}

fn de_stack<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    crate::spec::num::de_bounded_count(deserializer, "stack", MAX_STACK)
}

/// An image input expected by a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Image {
    pub role: String,
    /// The body part the camera the model wants sits on, when the role
    /// repeats across a body (`head`, `left_arm`, ...): an identity key the
    /// resolver matches on. A model that names one binds only that camera; one
    /// that names none binds the env's only camera of the role under any part.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub part: Option<String>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_count")]
    pub height: Option<u32>,
    #[serde(default, deserialize_with = "crate::spec::num::de_opt_count")]
    pub width: Option<u32>,
    #[serde(default)]
    pub layout: ImageLayout,
    /// Channel count the model expects (e.g. `3` for RGB, `1` for grayscale).
    /// When set, an env image with a different channel count is a resolve
    /// error: the adapter does not convert between channel counts.
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
    /// `[-1, 1]`).
    #[serde(default)]
    pub normalize: Normalize,
    #[serde(default, deserialize_with = "crate::spec::num::de_count")]
    pub lead_dims: u32,
    #[serde(default)]
    pub upside_down: bool,
    /// Resize algorithm the model's training pipeline used, one of
    /// [`RESAMPLES`](crate::v1::RESAMPLES), default `"bilinear"`. Un-suffixed
    /// names follow torch/OpenCV, `_aa` names PIL. An unknown name is a resolve
    /// error, not a parse failure; the value is always emitted.
    #[serde(default = "default_bilinear")]
    pub resample: String,
    /// Permit the resize to *upscale* (interpolate detail the env image does not
    /// have). Off by default: a target larger than the env's native resolution
    /// is then a resolve error.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub allow_upscale: bool,
    /// How to reconcile a target whose aspect ratio differs from the env image:
    /// one mode (`"stretch"`, `"crop"`, `"pad"`) or a preference list, of which
    /// the resolver picks, per env, the first that needs no disallowed upscale.
    /// Absent, an aspect-changing resize is a resolve error; an unrecognized
    /// mode is skipped.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit: Option<AcceptSet<FitMode>>,
    /// Zero-fill a black frame when the env does not provide this camera, instead
    /// of failing resolution. Needs `height`, `width` and `channels` to size the
    /// blank.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub optional: bool,
    /// Raw 8-bit fill level for an absent `optional` camera: `0` = black (the
    /// default), `255` = white, `128` ~ mid-gray. Applied before the
    /// normalize and dtype steps.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fill: Option<u8>,
    /// Number of consecutive observations the model stacks on a new leading
    /// axis (frame history); `1` = no stacking, at most `64`. The env still
    /// sends one frame per step; the window is kept per episode.
    #[serde(
        default = "crate::spec::num::default_one",
        deserialize_with = "de_stack",
        skip_serializing_if = "crate::spec::num::is_one"
    )]
    pub stack: u32,
    /// Side fraction of the frame a center crop keeps, in `(0, 1]` (`0.667`
    /// keeps the middle two thirds of each axis). Mutually exclusive with
    /// `crop_area`.
    #[serde(
        default,
        deserialize_with = "crate::spec::num::de_opt_unit_fraction",
        skip_serializing_if = "Option::is_none"
    )]
    pub crop: Option<f64>,
    /// Area fraction of the frame a center crop keeps, in `(0, 1]`; the side
    /// fraction is its square root (`0.9` -> `0.949`).
    #[serde(
        default,
        deserialize_with = "crate::spec::num::de_opt_unit_fraction",
        skip_serializing_if = "Option::is_none"
    )]
    pub crop_area: Option<f64>,
    /// How the crop box is taken, one of [`CROP_MODES`]. `"zoom"` (the default)
    /// resamples the fractional box straight to the target, as PIL's
    /// `Image.resize(size, box=...)`; `"slice"` cuts an integer center box
    /// first and resizes that.
    #[serde(default = "default_zoom", skip_serializing_if = "is_zoom")]
    pub crop_mode: String,
    /// Quality of a JPEG round-trip applied to the upright frame *before* the
    /// crop and resize, on the IJG `1..=100` scale (`95` is the usual training
    /// value). It reproduces the codec artifacts a pipeline that stored JPEG
    /// frames trained on (baseline, 4:2:0, standard IJG tables) and needs a
    /// 3-channel image.
    #[serde(
        default,
        deserialize_with = "crate::spec::num::de_opt_jpeg_quality",
        skip_serializing_if = "Option::is_none"
    )]
    pub jpeg_quality: Option<u8>,
    /// Channel order the model was trained on, one of [`CHANNEL_ORDERS`].
    /// `"bgr"` swaps red and blue after the spatial ops and before the dtype
    /// cast, and needs a 3-channel image.
    #[serde(default = "default_rgb", skip_serializing_if = "is_rgb")]
    pub channel_order: String,
    /// The camera resolution the model was trained against, as
    /// `[height, width]` (a bare integer on the wire is the square shorthand),
    /// each axis in `1..=4096`. An assertion, not a request: the core never
    /// resizes to reach it, and resolution fails with
    /// [`RenderMismatch`](crate::v1::ErrorCode::RenderMismatch) when the bound
    /// camera renders at another size. Callers can set the env's camera from
    /// it with [`render_requests`](crate::v1::render_requests).
    #[serde(
        default,
        deserialize_with = "crate::spec::num::de_opt_count_pair",
        skip_serializing_if = "Option::is_none"
    )]
    pub render: Option<(u32, u32)>,
    /// The frame window this stack gathers, as non-positive offsets from the
    /// current step, oldest first and ending at `0` (`[-6, -4, -2, 0]` is "every
    /// second frame of the last seven"). `None` is the contiguous window
    /// `stack` describes. When set its length must equal `stack`; resolution
    /// checks the offsets are non-positive, strictly increasing and end at `0`.
    #[serde(
        default,
        deserialize_with = "crate::spec::num::de_offsets",
        skip_serializing_if = "Option::is_none"
    )]
    pub offsets: Option<Vec<i32>>,
    /// What fills the window before an episode has produced enough frames:
    /// `"first"` (replicate the first frame, the default) or `"black"` (a raw
    /// 8-bit `0` frame through the plan).
    #[serde(default, skip_serializing_if = "StackPad::is_first")]
    pub stack_pad: StackPad,
    /// Unrecognized additive fields, retained for round-trip and surfaced to the
    /// publish-door `reject_unknowns` guard.
    #[serde(flatten)]
    pub unknown: BTreeMap<String, serde_json::Value>,
}

#[cfg(test)]
mod part_tests {
    use super::Image;

    #[test]
    fn part_is_omitted_when_unset_and_carried_when_set() {
        let image: Image = serde_json::from_str(r#"{"role": "image/wrist"}"#).expect("parse");
        assert_eq!(image.part, None);
        assert!(!serde_json::to_string(&image).unwrap().contains("part"));
        let image: Image =
            serde_json::from_str(r#"{"role": "image/wrist", "part": "left_arm"}"#).expect("parse");
        assert_eq!(image.part.as_deref(), Some("left_arm"));
        assert!(
            serde_json::to_string(&image)
                .unwrap()
                .contains(r#""part":"left_arm""#)
        );
    }
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
    fn fill_defaults_to_none_and_is_omitted_from_wire() {
        let input = image("");
        let ModelLeaf::Image(img) = &input else {
            panic!("expected image")
        };
        assert_eq!(img.fill, None);
        assert!(!serde_json::to_string(&input).unwrap().contains("fill"));
    }

    #[test]
    fn fill_roundtrips_when_set() {
        let input = image(r#", "fill": 128"#);
        let ModelLeaf::Image(img) = &input else {
            panic!("expected image")
        };
        assert_eq!(img.fill, Some(128));
        assert!(
            serde_json::to_string(&input)
                .unwrap()
                .contains("\"fill\":128")
        );
    }

    #[test]
    fn fill_rejects_out_of_range() {
        assert!(
            serde_json::from_str::<ModelLeaf>(
                r#"{"type": "image", "role": "image/primary", "fill": 300}"#
            )
            .is_err()
        );
    }

    #[test]
    fn normalize_overloads_bool_and_range() {
        use crate::spec::model::image::Normalize;

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
    fn crop_and_channel_order_default_off_and_stay_off_the_wire() {
        let input = image("");
        let ModelLeaf::Image(img) = &input else {
            panic!("expected image")
        };
        assert_eq!((img.crop, img.crop_area), (None, None));
        assert_eq!(
            (img.crop_mode.as_str(), img.channel_order.as_str()),
            ("zoom", "rgb")
        );
        let wire = serde_json::to_string(&input).unwrap();
        assert_eq!(img.jpeg_quality, None);
        for field in [
            "crop",
            "crop_area",
            "crop_mode",
            "channel_order",
            "jpeg_quality",
        ] {
            assert!(!wire.contains(field), "{field} leaked into {wire}");
        }
    }

    #[test]
    fn crop_fields_round_trip_when_set() {
        let input = image(r#", "crop_area": 0.9, "crop_mode": "slice", "channel_order": "bgr""#);
        let ModelLeaf::Image(img) = &input else {
            panic!("expected image")
        };
        assert_eq!(img.crop_area, Some(0.9));
        assert_eq!(img.crop_mode, "slice");
        assert_eq!(img.channel_order, "bgr");
        let wire = serde_json::to_string(&input).unwrap();
        assert!(wire.contains("\"crop_area\":0.9"), "got: {wire}");
        assert!(wire.contains("\"crop_mode\":\"slice\""), "got: {wire}");
        assert!(wire.contains("\"channel_order\":\"bgr\""), "got: {wire}");
    }

    #[test]
    fn crop_fraction_bounds_are_enforced_at_the_wire() {
        for bad in ["0", "-0.5", "1.5"] {
            let json = format!(r#"{{"type": "image", "role": "image/primary", "crop": {bad}}}"#);
            let err = serde_json::from_str::<ModelLeaf>(&json).expect_err("bounds");
            assert!(err.to_string().contains("fraction in (0, 1]"), "got: {err}");
        }
        assert!(
            serde_json::from_str::<ModelLeaf>(
                r#"{"type": "image", "role": "image/primary", "crop_area": 1}"#
            )
            .is_ok()
        );
    }

    #[test]
    fn jpeg_quality_round_trips_and_is_bounded_to_the_ijg_scale() {
        let input = image(r#", "jpeg_quality": 95"#);
        let ModelLeaf::Image(img) = &input else {
            panic!("expected image")
        };
        assert_eq!(img.jpeg_quality, Some(95));
        assert!(
            serde_json::to_string(&input)
                .unwrap()
                .contains("\"jpeg_quality\":95")
        );
        for bad in ["0", "101"] {
            let json =
                format!(r#"{{"type": "image", "role": "image/primary", "jpeg_quality": {bad}}}"#);
            let err = serde_json::from_str::<ModelLeaf>(&json).expect_err("bounds");
            assert!(err.to_string().contains("between 1 and 100"), "got: {err}");
        }
    }

    #[test]
    fn render_defaults_to_none_and_stays_off_the_wire() {
        let input = image("");
        let ModelLeaf::Image(img) = &input else {
            panic!("expected image")
        };
        assert_eq!(img.render, None);
        assert!(!serde_json::to_string(&input).unwrap().contains("render"));
    }

    #[test]
    fn render_accepts_a_square_int_and_a_pair_and_emits_a_pair() {
        for (declared, expected) in [("448", (448, 448)), ("[480, 640]", (480, 640))] {
            let input = image(&format!(r#", "render": {declared}"#));
            let ModelLeaf::Image(img) = &input else {
                panic!("expected image")
            };
            assert_eq!(img.render, Some(expected));
            let wire = serde_json::to_string(&input).unwrap();
            assert!(
                wire.contains(&format!("\"render\":[{},{}]", expected.0, expected.1)),
                "got: {wire}"
            );
        }
    }

    #[test]
    fn render_axis_bounds_are_enforced_at_the_wire() {
        for bad in ["0", "4097", "[0, 448]", "[448, 4097]"] {
            let json = format!(r#"{{"type": "image", "role": "image/primary", "render": {bad}}}"#);
            let err = serde_json::from_str::<ModelLeaf>(&json).expect_err("bounds");
            assert!(err.to_string().contains("between 1 and 4096"), "got: {err}");
        }
        let err = serde_json::from_str::<ModelLeaf>(
            r#"{"type": "image", "role": "image/primary", "render": [448]}"#,
        )
        .expect_err("length");
        assert!(
            err.to_string()
                .contains("an integer or a pair [height, width], got 1"),
            "got: {err}"
        );
        assert!(
            serde_json::from_str::<ModelLeaf>(
                r#"{"type": "image", "role": "image/primary", "render": [1, 4096]}"#
            )
            .is_ok()
        );
    }

    #[test]
    fn offsets_and_stack_pad_default_off_and_stay_off_the_wire() {
        let input = image("");
        let ModelLeaf::Image(img) = &input else {
            panic!("expected image")
        };
        assert_eq!(img.offsets, None);
        assert_eq!(img.stack_pad, super::StackPad::First);
        let wire = serde_json::to_string(&input).unwrap();
        for field in ["offsets", "stack_pad"] {
            assert!(!wire.contains(field), "{field} leaked into {wire}");
        }
    }

    #[test]
    fn offsets_and_stack_pad_round_trip_when_set() {
        let input = image(r#", "stack": 4, "offsets": [-6, -4, -2, 0], "stack_pad": "black""#);
        let ModelLeaf::Image(img) = &input else {
            panic!("expected image")
        };
        assert_eq!(img.offsets.as_deref(), Some(&[-6, -4, -2, 0][..]));
        assert_eq!(img.stack_pad, super::StackPad::Black);
        let wire = serde_json::to_string(&input).unwrap();
        assert!(wire.contains("\"offsets\":[-6,-4,-2,0]"), "got: {wire}");
        assert!(wire.contains("\"stack_pad\":\"black\""), "got: {wire}");
    }

    #[test]
    fn a_wrong_typed_offset_reads_in_domain_language() {
        let err = serde_json::from_str::<ModelLeaf>(
            r#"{"type": "image", "role": "image/primary", "offsets": [-2.5, 0]}"#,
        )
        .expect_err("element type");
        assert!(err.to_string().contains("a whole number"), "got: {err}");
        assert!(
            !err.to_string().contains("i32"),
            "leaks the wire type: {err}"
        );
        assert!(
            serde_json::from_str::<ModelLeaf>(
                r#"{"type": "image", "role": "image/primary", "offsets": [3, 1]}"#
            )
            .is_ok()
        );
    }

    #[test]
    fn tagged_payload_captures_unknown_field_for_round_trip() {
        let input = image(r#", "layuot": "chw""#);
        let ModelLeaf::Image(img) = &input else {
            panic!("expected image")
        };
        assert_eq!(img.layout, ImageLayout::Hwc);
        assert_eq!(img.unknown.get("layuot"), Some(&serde_json::json!("chw")));
        assert!(!img.unknown.contains_key("type"));
        assert!(serde_json::to_string(&input).unwrap().contains("layuot"));
    }
}
