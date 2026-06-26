//! Image axis layout vocabulary.

use serde::{Deserialize, Serialize};

/// Axis layout of an image value.
///
/// **Frozen v1 vocabulary** (same policy as [`crate::spec::RotationEncoding`]):
/// a new layout is a v2 key bump, not an additive v1 value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageLayout {
    #[default]
    Hwc,
    Chw,
}

impl ImageLayout {
    /// Every layout, for consumers exporting the vocabulary.
    pub const ALL: [Self; 2] = [Self::Hwc, Self::Chw];

    /// Wire/display name (matches the JSON form).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Hwc => "hwc",
            Self::Chw => "chw",
        }
    }
}

impl crate::spec::accept_set::WireVocab for ImageLayout {
    fn from_wire(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|layout| layout.as_str() == name)
    }

    fn as_wire(self) -> &'static str {
        self.as_str()
    }
}

/// How a resize reconciles a target whose aspect ratio differs from the source.
///
/// A model may declare a *preference list* of fit modes (an
/// [`AcceptSet`](crate::spec::AcceptSet)); the resolver picks, per env, the
/// first one that does not need a disallowed upscale — so the same spec can
/// crop a large camera and letterbox a small one. When the aspects already
/// match, every mode is the same uniform scale.
///
/// No `Default`: there is no default-construction path (unlike [`ImageLayout`],
/// whose `Default` backs `#[serde(default)]` on layout fields). The resolver
/// picks the `Stretch` fallback explicitly, so a derived default would be dead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitMode {
    /// Resize each axis independently to the target (distorts aspect).
    Stretch,
    /// Uniformly scale to cover the target, then center-crop (drops edges).
    Crop,
    /// Uniformly scale to fit within the target, then center-pad with zeros.
    Pad,
}

impl FitMode {
    /// Every fit mode, in the natural preference order (least surprising first).
    pub const ALL: [Self; 3] = [Self::Stretch, Self::Crop, Self::Pad];

    /// Wire/display name (matches the JSON form).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Stretch => "stretch",
            Self::Crop => "crop",
            Self::Pad => "pad",
        }
    }
}

impl crate::spec::accept_set::WireVocab for FitMode {
    fn from_wire(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|mode| mode.as_str() == name)
    }

    fn as_wire(self) -> &'static str {
        self.as_str()
    }
}

#[cfg(test)]
mod tests {
    use super::ImageLayout;

    #[test]
    fn known_vocab_roundtrips_and_serde_matches_as_str() {
        for layout in ImageLayout::ALL {
            let json = serde_json::to_string(&layout).expect("serialize");
            assert_eq!(json.trim_matches('"'), layout.as_str());
            let back: ImageLayout = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(back, layout);
        }
    }

    #[test]
    fn unknown_layout_is_rejected() {
        assert!(serde_json::from_str::<ImageLayout>("\"nhwc\"").is_err());
    }
}
