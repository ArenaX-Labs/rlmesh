//! Resolved instructions for one model state input.

use super::super::spec::{RotationEncoding, StateContainer};

/// One source slice feeding a resolved state input.
///
/// When `zero_fill` is set the piece has no env source: it contributes
/// `dim` zeros (an optional component the env did not declare).
#[derive(Debug, Clone, PartialEq)]
pub struct StatePiece {
    pub env_key: String,
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
