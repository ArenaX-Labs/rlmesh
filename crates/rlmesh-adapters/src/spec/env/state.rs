//! A numeric proprioception entry in an environment observation.
//!
//! Internal post-`join` form; never serialized (see `spec::env`), so no serde.

use crate::path::NodePath;
use crate::spec::AcceptSet;
use crate::spec::rotations::RotationEncoding;

/// A numeric proprioception entry in an environment observation.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvState {
    /// Structured source path into the raw observation tree this state is read
    /// from (the env-side placement); empty (root) for a bare single-leaf obs.
    pub source: NodePath,
    pub role: String,
    /// Start index of this feature within its space leaf, set only when it is
    /// one field of a [`SplitLayout`](crate::spec::env_tags::SplitLayout)
    /// slicing several role fields out of one flat numeric leaf. `None` for a
    /// whole-leaf state, which reads the entire runtime value (the space width
    /// in `dim` is advisory — used for resolve-time bounds checks, not runtime
    /// slicing).
    pub slice_offset: Option<u32>,
    pub dim: Option<u32>,
    /// Rotation encoding(s) this feature is declared in. As an env (producer)
    /// declaration the *first recognized* entry is the native (raw) encoding;
    /// any further entries are alternative representations it is willing to
    /// emit. A bare string on the wire for the common single-encoding case.
    pub encoding: Option<AcceptSet<RotationEncoding>>,
    pub range: Option<(f64, f64)>,
}
