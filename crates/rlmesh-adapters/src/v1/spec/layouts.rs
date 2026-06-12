//! Image axis layout vocabulary.

use serde::{Deserialize, Serialize};

/// Axis layout of an image value.
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
