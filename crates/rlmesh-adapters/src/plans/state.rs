//! Resolved instructions for one model state input.

use crate::path::NodePath;
use crate::spec::{FrameRef, RotationEncoding, RotationLiteral, StateContainer};

/// One source slice feeding a resolved state input.
///
/// When `fill` is set the piece has no env source: it contributes `dim` copies
/// of the fill value — a declared constant part, or (with `absent_role`) an
/// optional component the env did not declare.
#[derive(Debug, Clone, PartialEq)]
pub struct StatePiece {
    /// Where this piece is read from in the raw observation tree (empty when
    /// `fill` is set).
    pub source: NodePath,
    /// Start index of the env feature within its space leaf, set only when the
    /// feature is one field of a flat-leaf `SplitLayout`: the leaf's runtime
    /// values are sliced to `[src_offset, src_offset + src_dim)` before any
    /// conversion. `None` reads the whole leaf (a non-layout state).
    pub src_offset: Option<u32>,
    /// Width of the env field's slice, used only when `src_offset` is set.
    pub src_dim: Option<u32>,
    pub src_encoding: Option<RotationEncoding>,
    pub dst_encoding: Option<RotationEncoding>,
    /// Fixed rotation right-multiplied onto the decoded source rotation before
    /// it is re-encoded (`R_out = R_in @ R(post_rotate)`).
    pub post_rotate: Option<RotationLiteral>,
    pub dim: Option<u32>,
    pub index: Option<u32>,
    /// Source value range (the env feature's), mapped into `dst_range`.
    pub src_range: Option<(f64, f64)>,
    /// Target value range (the model component's).
    pub dst_range: Option<(f64, f64)>,
    /// Model-side affine applied after the range map: `value * scale + offset`.
    pub scale: Option<f64>,
    pub offset: Option<f64>,
    /// The constant this piece contributes instead of reading the env, with
    /// `scale`/`offset` already folded in. `None` means a real env source.
    pub fill: Option<f64>,
    /// Whether a set `fill` stands in for an *absent* optional role (fabricated
    /// data the fit report confesses) rather than a declared constant part.
    pub absent_role: bool,
    /// The agreed coordinate frame this piece's values are in, when either side
    /// declared one (`None` when both were silent). Rendered by `describe`; the
    /// disagreement it would represent is already a resolve error.
    pub frame: Option<FrameRef>,
    /// The body part this piece was bound under, when a side declared one:
    /// the model's own `part`, else the env leaf's. Rendered by `describe` as
    /// `#part`; `None` when neither side named one, so every pre-`part`
    /// summary is unchanged.
    pub part: Option<String>,
    /// Resolved output width of this piece, when statically known (`None` when
    /// the env feature declares no width and nothing else fixes it). A
    /// host-side custom encoding addresses its own slice of a multi-part state
    /// by these widths, and `apply_state` asserts every piece against its width
    /// so a runtime value of another width is a loud error rather than a
    /// silently shifted layout.
    pub width: Option<u32>,
}

/// Resolved instructions for one model state input.
#[derive(Debug, Clone, PartialEq)]
pub struct StatePlan {
    /// Where this state lands in the assembled payload tree.
    pub placement: NodePath,
    pub pieces: Vec<StatePiece>,
    pub pad_to: Option<u32>,
    /// Assembled width before `pad_to`, when every piece's width is known.
    pub native_width: Option<u32>,
    pub dtype: String,
    pub reshape: Option<Vec<i64>>,
    pub container: StateContainer,
}
