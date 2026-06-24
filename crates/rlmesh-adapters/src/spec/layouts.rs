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
