//! A numeric proprioception entry in an environment observation.

use serde::{Deserialize, Serialize};

use crate::spec::rotations::RotationEncoding;

/// A numeric proprioception entry in an environment observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvState {
    pub key: String,
    pub role: String,
    /// Start index of this feature within its space leaf, set only when it is
    /// one field of a [`StateLayout`](crate::spec::env_tags::StateLayout)
    /// slicing several role fields out of one flat numeric leaf. `None` for a
    /// whole-leaf state, which reads the entire runtime value (the space width
    /// in `dim` is advisory — used for resolve-time bounds checks, not runtime
    /// slicing).
    #[serde(default)]
    pub slice_offset: Option<u32>,
    #[serde(default)]
    pub dim: Option<u32>,
    #[serde(default)]
    pub encoding: Option<RotationEncoding>,
    #[serde(default)]
    pub range: Option<(f64, f64)>,
}
