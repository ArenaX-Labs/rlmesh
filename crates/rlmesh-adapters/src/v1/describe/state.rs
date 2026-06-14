//! Summarize a resolved state plan.

use std::fmt::Write as _;

use super::super::fmt::{quoted, quoted_range};
use super::super::plans::StatePlan;

pub(super) fn describe_state(plan: &StatePlan) -> String {
    let mut parts: Vec<String> = Vec::new();
    for piece in &plan.pieces {
        if piece.zero_fill {
            parts.push(format!(
                "zeros({})",
                piece.dim.expect("zero-fill pieces always carry a width")
            ));
            continue;
        }
        let mut note = piece.env_key.clone();
        // A StateLayout field reads a fixed `[offset, offset+width)` slice of a
        // flat leaf; show it so the split is visible. A whole-leaf state leaves
        // src_offset None and reads the entire value.
        if let Some(offset) = piece.src_offset {
            let width = piece.src_dim.expect("layout fields carry src_dim");
            let _ = write!(note, "[{offset}:{}]", offset + width);
        }
        if piece.src_encoding != piece.dst_encoding {
            let src = piece
                .src_encoding
                .expect("conversions require both encodings");
            let dst = piece
                .dst_encoding
                .expect("conversions require both encodings");
            let _ = write!(note, " ({}->{})", src.as_str(), dst.as_str());
        }
        if let Some(index) = piece.index {
            let _ = write!(note, "[{index}]");
        } else if let Some(dim) = piece.dim {
            // The env slice above already states a layout field's width; only
            // note a model-side truncation when it narrows that slice further.
            if piece.src_dim != Some(dim) {
                let _ = write!(note, "[:{dim}]");
            }
        }
        if let (Some(src), Some(dst)) = (piece.src_range, piece.dst_range) {
            let _ = write!(
                note,
                " (range {}->{})",
                quoted_range(src),
                quoted_range(dst)
            );
        }
        parts.push(note);
    }
    let suffix = match plan.pad_to {
        Some(pad_to) => format!(", pad to {pad_to}"),
        None => String::new(),
    };
    format!(
        "{} <- concat({}){}",
        quoted(&plan.model_key),
        parts.join(", "),
        suffix
    )
}
