//! Summarize a resolved action segment.

use std::fmt::Write as _;

use super::super::plans::ActionSegment;
use super::super::pyfmt::{py_repr, py_repr_range};

/// Summarize how one env action component is derived from the model output.
pub fn describe_segment(segment: &ActionSegment) -> String {
    let mut note = format!(
        "{} <- model[{}:{}]",
        py_repr(&segment.role),
        segment.start,
        segment.stop
    );
    if segment.src_encoding != segment.dst_encoding {
        let src = segment
            .src_encoding
            .expect("conversions require both encodings");
        let dst = segment
            .dst_encoding
            .expect("conversions require both encodings");
        let _ = write!(note, " ({}->{})", src.as_str(), dst.as_str());
    }
    if let (Some(src_range), Some(dst_range)) = (segment.src_range, segment.dst_range) {
        let _ = write!(
            note,
            " (range {}->{})",
            py_repr_range(src_range),
            py_repr_range(dst_range)
        );
    }
    if segment.binarize {
        note.push_str(" (sign)");
    }
    note
}
