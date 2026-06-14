//! Resolved instructions for one model state input.

use crate::spec::{RotationEncoding, StateContainer};

/// One source slice feeding a resolved state input.
///
/// When `zero_fill` is set the piece has no env source: it contributes
/// `dim` zeros (an optional component the env did not declare).
#[derive(Debug, Clone, PartialEq)]
pub struct StatePiece {
    pub env_key: String,
    /// Start index of the env feature within its space leaf, set only when the
    /// feature is one field of a flat-leaf `StateLayout`: the leaf's runtime
    /// values are sliced to `[src_offset, src_offset + src_dim)` before any
    /// conversion. `None` reads the whole leaf (a non-layout state).
    pub src_offset: Option<u32>,
    /// Width of the env field's slice, used only when `src_offset` is set.
    pub src_dim: Option<u32>,
    pub src_encoding: Option<RotationEncoding>,
    pub dst_encoding: Option<RotationEncoding>,
    pub dim: Option<u32>,
    pub index: Option<u32>,
    /// Source value range (the env feature's), mapped into `dst_range`.
    pub src_range: Option<(f64, f64)>,
    /// Target value range (the model component's).
    pub dst_range: Option<(f64, f64)>,
    pub zero_fill: bool,
}

/// Resolved instructions for one model state input.
#[derive(Debug, Clone, PartialEq)]
pub struct StatePlan {
    pub model_key: String,
    pub pieces: Vec<StatePiece>,
    pub pad_to: Option<u32>,
    pub dtype: String,
    pub reshape: Option<Vec<i64>>,
    pub container: StateContainer,
}
